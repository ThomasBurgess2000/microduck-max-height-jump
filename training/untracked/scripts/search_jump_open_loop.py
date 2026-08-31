#!/usr/bin/env python3
"""CEM search for Microduck's nominal open-loop bilateral jump ceiling.

The search perturbs a known lifting checkpoint's action trace instead of
guessing poses that the voltage-limited BAM actuators may never reach.  It
scores the same V5 physical quantity used by training -- min(CoM rise,
bilateral sole gap) -- while retaining the honest BAM voltage, delay, friction,
and torque model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg


PARAM_NAMES = (
    "time_scale",
    "leg_gain",
    "launch_gain",
    "head_gain",
    "hip_roll_gain",
    "knee_bias",
    "ankle_bias",
    "recovery_gain",
)
LOW = torch.tensor((0.75, 0.70, 0.70, 0.40, 0.45, -0.75, -0.60, 0.25))
HIGH = torch.tensor((1.50, 1.70, 2.00, 1.50, 1.25, 0.75, 0.60, 1.30))
NEUTRAL = torch.tensor((1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0))


def _nominal_cfg(num_envs: int):
    cfg = make_microduck_jump_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.curriculum.clear()
    cfg.terminations.clear()
    keep = {
        "reset_base",
        "reset_robot_joints",
        "reset_action_history",
        "set_jump_state",
        "expand_bam_friction_fields",
    }
    cfg.events = {name: term for name, term in cfg.events.items() if name in keep}
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.vin_range = (7.4, 7.4)
    actuator.vin_drop_gain_range = (0.1, 0.1)
    actuator.delay_min_lag = 4
    actuator.delay_max_lag = 4
    return cfg


def _load_anchor(path: Path, max_steps: int) -> torch.Tensor:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or invalid trace: {path}")
    actions = torch.tensor([row["action"] for row in rows[:max_steps]], dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"expected trace actions shaped (T, 14), got {tuple(actions.shape)}")
    return actions


def _anchor_action(
    anchor: torch.Tensor,
    params: torch.Tensor,
    step: int,
) -> torch.Tensor:
    n = params.shape[0]
    time_scale = params[:, 0]
    anchor_idx = torch.floor(step / time_scale).long()
    active = anchor_idx < anchor.shape[0]
    clamped_idx = torch.clamp(anchor_idx, max=anchor.shape[0] - 1)
    action = anchor[clamped_idx].clone()

    leg_ids = torch.tensor((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), device=action.device)
    action[:, leg_ids] *= params[:, 1:2]
    # The seed's physical launch spans source steps 5..12.
    launching = (anchor_idx >= 4) & (anchor_idx <= 11)
    action[:, leg_ids] *= torch.where(launching, params[:, 2], torch.ones_like(time_scale))[:, None]
    action[:, 5:9] *= params[:, 3:4]
    action[:, (1, 10)] *= params[:, 4:5]
    action[:, 3] -= params[:, 5]
    action[:, 12] += params[:, 5]
    action[:, 4] += params[:, 6]
    action[:, 13] -= params[:, 6]
    recovering = anchor_idx >= 12
    action *= torch.where(recovering, params[:, 7], torch.ones_like(time_scale))[:, None]
    return torch.where(active[:, None], action, torch.zeros((n, 14), device=action.device))


def _run(
    env: ManagerBasedRlEnv,
    params: torch.Tensor,
    anchor: torch.Tensor,
    *,
    writer=None,
) -> dict[str, torch.Tensor]:
    env.reset()
    n = env.num_envs
    device = env.device
    params = params.to(device)
    anchor = anchor.to(device)
    total_steps = round(anchor.shape[0] * float(HIGH[0])) + 45

    max_tilt = torch.zeros(n, device=device)
    max_drift = torch.zeros(n, device=device)
    any_body_contact = torch.zeros(n, dtype=torch.bool, device=device)
    initial_xy = env.scene["robot"].data.root_com_pos_w[:, :2].clone()
    for step in range(total_steps):
        action = _anchor_action(anchor, params, step)
        env.step(action)
        asset = env.scene["robot"]
        mdp._update_jump_state(env, asset)
        quat = asset.data.root_link_quat_w
        cos_tilt = torch.clamp(
            1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0
        )
        flight = env._jump_takeoff_latched & ~env._jump_landed_latched
        flight_tilt = torch.where(flight, torch.acos(cos_tilt), torch.zeros_like(cos_tilt))
        max_tilt = torch.maximum(max_tilt, flight_tilt)
        flight_drift = torch.where(
            flight[:, None],
            asset.data.root_com_pos_w[:, :2] - initial_xy,
            torch.zeros_like(initial_xy),
        )
        max_drift = torch.maximum(
            max_drift,
            torch.norm(flight_drift, dim=-1),
        )
        any_body_contact |= (
            mdp._jump_contact_mask(env, "nonfoot_ground_contact").any(dim=-1)
            & ~env._jump_landed_latched
        )
        if writer is not None:
            writer.append_data(env.render())

    body_contact = any_body_contact
    visible = env._jump_visible_height_frontier
    # Ceiling search allows an imperfect landing, but rejects launch-time body
    # strikes and strongly tilted ballistic flings.
    score = (
        visible
        - 0.001 * torch.clamp(max_tilt - torch.deg2rad(torch.tensor(25.0, device=device)), min=0.0)
        - 0.05 * max_drift
        - 0.020 * body_contact.float()
    )
    return {
        "score": score,
        "visible_height": visible,
        "clearance": env._jump_clearance_frontier,
        "com_rise": env._jump_height_frontier,
        "launch_velocity": env._jump_launch_frontier,
        "airtime": env._jump_airtime_frontier,
        "max_tilt": max_tilt,
        "body_contact": body_contact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--elite-frac", type=float, default=0.10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--anchor-trace", type=Path, required=True)
    parser.add_argument("--anchor-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.video is not None and args.video.exists():
        raise FileExistsError(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)

    configure_torch_backends()
    torch.manual_seed(args.seed)
    anchor = _load_anchor(args.anchor_trace, args.anchor_steps)
    device = torch.device(args.device)
    low, high = LOW.to(device), HIGH.to(device)
    mean = NEUTRAL.to(device).clone()
    std = (high - low) / 3.0
    env = ManagerBasedRlEnv(cfg=_nominal_cfg(args.population), device=args.device)

    elite_count = max(2, round(args.population * args.elite_frac))
    best_score = -float("inf")
    best_params = mean.clone()
    best_metrics: dict[str, float | bool] = {}
    history = []
    for iteration in range(args.iterations):
        samples = torch.clamp(
            mean + std * torch.randn((args.population, len(PARAM_NAMES)), device=device),
            low,
            high,
        )
        samples[0] = NEUTRAL.to(device)
        samples[1] = mean
        metrics = _run(env, samples, anchor)
        elite_ids = torch.topk(metrics["score"], elite_count).indices
        elite = samples[elite_ids]
        mean = elite.mean(dim=0)
        std = torch.maximum(elite.std(dim=0, unbiased=False), (high - low) * 0.02)
        idx = int(torch.argmax(metrics["score"]).item())
        score = float(metrics["score"][idx].item())
        if score > best_score:
            best_score = score
            best_params = samples[idx].clone()
            best_metrics = {
                name: (
                    bool(value[idx].item())
                    if value.dtype == torch.bool
                    else float(value[idx].item())
                )
                for name, value in metrics.items()
                if name != "score"
            }
        row = {
            "iteration": iteration,
            "best_visible_height_mm": float(metrics["visible_height"].max().item() * 1000.0),
            "elite_mean_visible_height_mm": float(metrics["visible_height"][elite_ids].mean().item() * 1000.0),
            "best_score": float(metrics["score"].max().item()),
        }
        history.append(row)
        print(json.dumps(row))
    env.close()

    result = {
        "seed": args.seed,
        "device": args.device,
        "population": args.population,
        "iterations": args.iterations,
        "parameter_names": PARAM_NAMES,
        "best_params": {
            name: float(value)
            for name, value in zip(PARAM_NAMES, best_params.cpu().tolist(), strict=True)
        },
        "best_score": best_score,
        "best_metrics": {
            name: (
                value
                if isinstance(value, bool)
                else value
                * (1000.0 if name in {"visible_height", "clearance", "com_rise"} else 1.0)
            )
            for name, value in best_metrics.items()
        },
        "metric_units": {
            "visible_height": "mm",
            "clearance": "mm",
            "com_rise": "mm",
            "launch_velocity": "m/s",
            "airtime": "s",
            "max_tilt": "rad",
        },
        "history": history,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    if args.video is not None:
        video_env = ManagerBasedRlEnv(
            cfg=_nominal_cfg(1), device=args.device, render_mode="rgb_array"
        )
        with imageio.get_writer(args.video, fps=50, codec="libx264") as writer:
            video_metrics = _run(video_env, best_params[None, :], anchor, writer=writer)
        video_env.close()
        result["video_metrics"] = {
            name: (
                bool(value[0].item())
                if value.dtype == torch.bool
                else float(value[0].item())
            )
            for name, value in video_metrics.items()
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"video={args.video.resolve()}")
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
