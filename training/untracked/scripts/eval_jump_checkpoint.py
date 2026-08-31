#!/usr/bin/env python3
"""Evaluate one jump checkpoint, export ONNX, and record a side-profile video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


TASK_ID = "Mjlab-Jump-Flat-MicroDuck"


def _any_contact(env: ManagerBasedRlEnv, sensor_name: str) -> bool:
    found = env.scene.sensors[sensor_name].data.found
    return bool(found.reshape(found.shape[0], -1)[0].bool().any().item())


def _all_primary_contacts(env: ManagerBasedRlEnv, sensor_name: str) -> bool:
    """Whether every primary geom in a contact sensor is touching."""
    found = env.scene.sensors[sensor_name].data.found
    if found.ndim > 2:
        found = found.reshape(found.shape[0], found.shape[1], -1).any(dim=-1)
    elif found.ndim == 1:
        found = found[:, None]
    return bool(found[0].bool().all().item())


def _actor_observation(observations: object) -> torch.Tensor:
    """Unwrap mjlab/rsl_rl's nested actor TensorDict into its 61D tensor."""
    actor = observations
    for _ in range(3):
        if isinstance(actor, torch.Tensor):
            return actor
        try:
            actor = actor["actor"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise TypeError(
                f"could not extract actor tensor from {type(observations).__name__}"
            ) from exc
    raise TypeError(
        f"actor observation did not resolve to a tensor: {type(actor).__name__}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument(
        "--policy-onnx",
        type=Path,
        help="Evaluate an existing baked-normalizer policy when its old checkpoint is incompatible",
    )
    parser.add_argument(
        "--checkpoint-nonstrict",
        action="store_true",
        help="Ignore obsolete training-only distribution keys in legacy deterministic checkpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--trace",
        type=Path,
        help="Optional per-control-step JSON trace including policy actions and clearance",
    )
    parser.add_argument(
        "--slow-output",
        type=Path,
        help="Optional full-frame slow-motion MP4 generated from the truthful rollout",
    )
    parser.add_argument(
        "--slow-foot-output",
        type=Path,
        help="Optional 4x slow-motion lower-body close-up MP4",
    )
    parser.add_argument("--slow-factor", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=174, help="174 steps = 3.48 s at 50 Hz")
    parser.add_argument(
        "--preroll-s",
        type=float,
        default=0.0,
        help="Seconds to hold the initial reset frame before policy inference",
    )
    parser.add_argument(
        "--postroll-s",
        type=float,
        default=0.0,
        help="Seconds to hold the terminal landing frame after inference",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--standing-policy",
        type=Path,
        help="Optional baked-normalizer 61D standing-policy ONNX used after landing",
    )
    parser.add_argument(
        "--handoff-start-s",
        type=float,
        default=0.36,
        help="Policy time at which the smooth jump-to-standing blend begins",
    )
    parser.add_argument(
        "--handoff-blend-s",
        type=float,
        default=0.20,
        help="Duration of the smoothstep jump-to-standing action blend",
    )
    parser.add_argument(
        "--continue-after-failed-recovery",
        action="store_true",
        help="Keep the demo running past the jump task's 1.4 s recovery cutoff",
    )
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.policy_onnx is None):
        parser.error("provide exactly one checkpoint or --policy-onnx")
    policy_source = args.checkpoint or args.policy_onnx
    assert policy_source is not None
    if not policy_source.exists():
        raise FileNotFoundError(policy_source)
    if args.standing_policy is not None and not args.standing_policy.exists():
        raise FileNotFoundError(args.standing_policy)
    if args.handoff_blend_s <= 0.0:
        raise ValueError("--handoff-blend-s must be positive")
    requested_outputs = [args.output, args.onnx, args.metrics]
    if args.trace is not None:
        requested_outputs.append(args.trace)
    requested_outputs.extend(
        path for path in (args.slow_output, args.slow_foot_output) if path is not None
    )
    for path in requested_outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 7
    # Preserve the terminal landing pose for both measurement and recording.
    # Auto-reset would replace it with the next episode's spawn in the same step.
    env_cfg.auto_reset = False
    failed_recovery_cfg = env_cfg.terminations.get("jump_failed_recovery")
    recovery_timeout_s = (
        float(failed_recovery_cfg.params.get("recovery_timeout", 0.80))
        if failed_recovery_cfg is not None
        else 0.80
    )
    if args.continue_after_failed_recovery:
        env_cfg.terminations.pop("jump_failed_recovery", None)
    agent_cfg = load_rl_cfg(TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    recording_dir = args.output.parent / f".{args.output.stem}_recording"
    render_fps = int(raw_env.metadata.get("render_fps", 50))
    preroll_frames = max(0, round(args.preroll_s * render_fps))
    postroll_frames = max(0, round(args.postroll_s * render_fps))
    video_recorder = VideoRecorder(
        raw_env,
        video_folder=recording_dir,
        step_trigger=None,
        video_length=args.steps + preroll_frames + postroll_frames,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(video_recorder, clip_actions=agent_cfg.clip_actions)
    if args.policy_onnx is None:
        assert args.checkpoint is not None
        runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(args.checkpoint),
            strict=not args.checkpoint_nonstrict,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)

        # Mandatory export path: runner export includes the empirical actor
        # normalizer, then the deployment metadata is attached.
        runner.export_policy_to_onnx(str(args.onnx.parent), args.onnx.name)
        metadata = get_base_metadata(env.unwrapped, run_path=str(args.checkpoint))
        attach_metadata_to_onnx(str(args.onnx), metadata)
    else:
        jump_session = ort.InferenceSession(
            str(args.policy_onnx), providers=["CPUExecutionProvider"]
        )
        jump_input = jump_session.get_inputs()[0]
        jump_output = jump_session.get_outputs()[0]
        if jump_input.shape != [1, 61] or jump_output.shape != [1, 14]:
            raise ValueError(
                "jump policy must have [1, 61] input and [1, 14] output; "
                f"got {jump_input.shape} and {jump_output.shape}"
            )
        shutil.copy2(args.policy_onnx, args.onnx)

        def policy(observations: object) -> torch.Tensor:
            actor_obs = _actor_observation(observations)
            action_np = jump_session.run(
                None,
                {
                    jump_input.name: actor_obs.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                },
            )[0]
            return torch.from_numpy(action_np).to(
                device=actor_obs.device, dtype=actor_obs.dtype
            )
    standing_session = None
    if args.standing_policy is not None:
        standing_session = ort.InferenceSession(
            str(args.standing_policy), providers=["CPUExecutionProvider"]
        )
        standing_input = standing_session.get_inputs()[0]
        standing_output = standing_session.get_outputs()[0]
        if standing_input.shape != [1, 61] or standing_output.shape != [1, 14]:
            raise ValueError(
                "standing policy must have [1, 61] input and [1, 14] output; "
                f"got {standing_input.shape} and {standing_output.shape}"
            )

    initial_two_foot_contact = _all_primary_contacts(
        env.unwrapped, "feet_ground_contact"
    )
    video_recorder.trigger_type = "step"
    video_recorder._start_recording()
    for _ in range(preroll_frames):
        video_recorder._record_frame()

    stats = {
        "checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "policy_onnx": (
            str(args.policy_onnx.resolve()) if args.policy_onnx is not None else None
        ),
        "standing_policy": (
            str(args.standing_policy.resolve())
            if args.standing_policy is not None
            else None
        ),
        "handoff_start_s": args.handoff_start_s if standing_session is not None else None,
        "handoff_blend_s": args.handoff_blend_s if standing_session is not None else None,
        "continued_after_failed_recovery": args.continue_after_failed_recovery,
        "recovery_timeout_s": recovery_timeout_s,
        "stable_recovery_margin_ms": None,
        "initial_two_foot_contact": initial_two_foot_contact,
        "first_two_foot_contact_step": None,
        "qualified_takeoff": False,
        "raw_contact_loss_step": None,
        "takeoff_step": None,
        "max_launch_velocity_m_s": 0.0,
        "max_com_rise_mm": 0.0,
        "max_bilateral_sole_clearance_mm": 0.0,
        "max_visible_jump_height_mm": 0.0,
        "max_airtime_ms": 0.0,
        "touchdown": False,
        "touchdown_step": None,
        "stable_landing": False,
        "durable_landing": False,
        "stable_step": None,
        "failed_recovery": False,
        "max_settle_score": 0.0,
        "post_touchdown_survival_s": 0.0,
        "nonfoot_body_contact": False,
        "final_tilt_deg": None,
        "final_servo_joint_error_rad": None,
        "final_head_joint_error_rad": None,
        "final_foot_distance_mm": None,
        "minimum_post_touchdown_foot_distance_mm": None,
        "steps": 0,
    }
    obs = env.get_observations()
    trace = []
    robot = env.unwrapped.scene["robot"]
    servo_ids, servo_names = robot.find_joints(r"^(?!passive_).*")
    foot_ids, _ = robot.find_sites(r"^(left_foot|right_foot)$")
    head_indices = [
        i
        for i, name in enumerate(servo_names)
        if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
    ]
    previous_com_z = float(env.unwrapped.scene["robot"].data.root_com_pos_w[0, 2].item())
    with torch.inference_mode():
        for step in range(args.steps):
            jump_actions = policy(obs)
            standing_actions = None
            blend_alpha = 0.0
            if standing_session is not None:
                actor_obs = _actor_observation(obs)
                standing_np = standing_session.run(
                    None,
                    {
                        standing_session.get_inputs()[0].name: actor_obs.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    },
                )[0]
                standing_actions = torch.from_numpy(standing_np).to(
                    device=jump_actions.device, dtype=jump_actions.dtype
                )
                policy_time_s = step * env.unwrapped.step_dt
                blend_u = min(
                    1.0,
                    max(
                        0.0,
                        (policy_time_s - args.handoff_start_s)
                        / args.handoff_blend_s,
                    ),
                )
                blend_alpha = blend_u * blend_u * (3.0 - 2.0 * blend_u)
            actions = (
                torch.lerp(jump_actions, standing_actions, blend_alpha)
                if standing_actions is not None
                else jump_actions
            )
            obs, _, dones, extras = env.step(actions)
            base = env.unwrapped
            done = bool(dones[0].item())

            stats["steps"] = step + 1
            two_foot_contact = _all_primary_contacts(base, "feet_ground_contact")
            if stats["first_two_foot_contact_step"] is None and two_foot_contact:
                stats["first_two_foot_contact_step"] = step + 1
            if stats["raw_contact_loss_step"] is None and bool(
                base._jump_takeoff_latched[0].item()
            ):
                stats["raw_contact_loss_step"] = step + 1
            if stats["takeoff_step"] is None and bool(
                base._jump_valid_takeoff_latched[0].item()
            ):
                stats["takeoff_step"] = step + 1
            if stats["touchdown_step"] is None and bool(
                base._jump_touchdown_pulse[0].item()
            ):
                stats["touchdown_step"] = step + 1
            if stats["stable_step"] is None and bool(
                base._jump_stable_pulse[0].item()
            ):
                stats["stable_step"] = step + 1
            stats["qualified_takeoff"] |= bool(
                base._jump_valid_takeoff_latched[0].item()
            )
            stats["touchdown"] |= bool(base._jump_landed_latched[0].item())
            stats["stable_landing"] |= bool(base._jump_stable_latched[0].item())
            stats["durable_landing"] |= bool(base._jump_durable_success[0].item())
            stats["failed_recovery"] |= bool(
                base._jump_landed_latched[0].item()
                and not base._jump_stable_latched[0].item()
                and base._jump_post_touchdown_steps[0].item()
                >= int(math.ceil(recovery_timeout_s / base.step_dt))
            )
            stats["max_settle_score"] = max(
                stats["max_settle_score"],
                float(getattr(base, "_jump_settle_frontier")[0].item()),
            )
            stats["max_launch_velocity_m_s"] = max(
                stats["max_launch_velocity_m_s"], float(base._jump_launch_frontier[0].item())
            )
            # Independent rollout measurement.  Successful terminal steps are
            # auto-reset inside env.step, so exclude that reset discontinuity.
            current_com_z = float(base.scene["robot"].data.root_com_pos_w[0, 2].item())
            if not done:
                stats["max_launch_velocity_m_s"] = max(
                    stats["max_launch_velocity_m_s"],
                    (current_com_z - previous_com_z) / base.step_dt,
                )
                previous_com_z = current_com_z
            stats["max_com_rise_mm"] = max(
                stats["max_com_rise_mm"], float(base._jump_height_frontier[0].item()) * 1000.0
            )
            stats["max_bilateral_sole_clearance_mm"] = max(
                stats["max_bilateral_sole_clearance_mm"],
                float(base._jump_clearance_frontier[0].item()) * 1000.0,
            )
            stats["max_visible_jump_height_mm"] = max(
                stats["max_visible_jump_height_mm"],
                float(base._jump_visible_height_frontier[0].item()) * 1000.0,
            )
            stats["max_airtime_ms"] = max(
                stats["max_airtime_ms"], float(base._jump_airtime_frontier[0].item()) * 1000.0
            )
            stats["nonfoot_body_contact"] |= _any_contact(base, "nonfoot_ground_contact")
            stats["post_touchdown_survival_s"] = max(
                stats["post_touchdown_survival_s"],
                float(base._jump_post_touchdown_steps[0].item()) * base.step_dt,
            )
            quat = base.scene["robot"].data.root_link_quat_w[0]
            cos_tilt = torch.clamp(
                1.0 - 2.0 * (quat[1].square() + quat[2].square()), -1.0, 1.0
            )
            stats["final_tilt_deg"] = math.degrees(math.acos(float(cos_tilt.item())))
            servo_pos = robot.data.joint_pos[0, servo_ids]
            servo_default = robot.data.default_joint_pos[0, servo_ids]
            servo_error = servo_pos - servo_default
            head_error = servo_error[head_indices]
            foot_xy = robot.data.site_pos_w[0, foot_ids, :2]
            foot_distance_mm = float(torch.norm(foot_xy[0] - foot_xy[1]).item() * 1000.0)
            stats["final_servo_joint_error_rad"] = {
                name: float(error)
                for name, error in zip(servo_names, servo_error.detach().cpu().tolist(), strict=True)
            }
            stats["final_head_joint_error_rad"] = {
                servo_names[i]: float(servo_error[i].item()) for i in head_indices
            }
            stats["final_foot_distance_mm"] = foot_distance_mm
            if bool(base._jump_landed_latched[0].item()):
                previous_minimum = stats["minimum_post_touchdown_foot_distance_mm"]
                stats["minimum_post_touchdown_foot_distance_mm"] = (
                    foot_distance_mm
                    if previous_minimum is None
                    else min(previous_minimum, foot_distance_mm)
                )
            trace.append(
                {
                    "step": step + 1,
                    "time_s": (step + 1) * base.step_dt,
                    "policy_source": (
                        "jump"
                        if standing_session is None or blend_alpha <= 0.0
                        else "standing"
                        if blend_alpha >= 1.0
                        else "blend"
                    ),
                    "standing_blend_alpha": blend_alpha,
                    "action": [float(x) for x in actions[0].detach().cpu().tolist()],
                    "jump_action": [
                        float(x) for x in jump_actions[0].detach().cpu().tolist()
                    ],
                    "standing_action": (
                        [
                            float(x)
                            for x in standing_actions[0].detach().cpu().tolist()
                        ]
                        if standing_actions is not None
                        else None
                    ),
                    "two_foot_contact": two_foot_contact,
                    "raw_contact_loss": bool(base._jump_takeoff_latched[0].item()),
                    "qualified_takeoff": bool(
                        base._jump_valid_takeoff_latched[0].item()
                    ),
                    "bilateral_clearance_mm": float(
                        base._jump_clearance[0].item() * 1000.0
                    ),
                    "visible_height_mm": float(
                        base._jump_visible_height_frontier[0].item() * 1000.0
                    ),
                    "com_vz_m_s": float(base._jump_com_vz[0].item()),
                    "tilt_deg": stats["final_tilt_deg"],
                    "body_contact": _any_contact(base, "nonfoot_ground_contact"),
                    "servo_joint_position_rad": {
                        name: float(value)
                        for name, value in zip(
                            servo_names, servo_pos.detach().cpu().tolist(), strict=True
                        )
                    },
                    "servo_joint_error_rad": {
                        name: float(error)
                        for name, error in zip(
                            servo_names, servo_error.detach().cpu().tolist(), strict=True
                        )
                    },
                    "foot_xy_m": {
                        "left": [float(x) for x in foot_xy[0].detach().cpu().tolist()],
                        "right": [float(x) for x in foot_xy[1].detach().cpu().tolist()],
                    },
                    "foot_distance_mm": foot_distance_mm,
                }
            )

            # Recover the terminal reason from the episode log before stopping
            # the first rollout.
            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            success_metric = log.get("Episode_Termination/jump_success", 0.0)
            if isinstance(success_metric, torch.Tensor):
                success_metric = float(success_metric.max().item())
            stats["stable_landing"] |= float(success_metric) > 0.0
            if done:
                break

    # auto_reset=False leaves the genuine terminal landing state in MuJoCo.
    # Hold it long enough for the landing result to be visually inspectable.
    for _ in range(postroll_frames):
        video_recorder._record_frame()

    if stats["touchdown_step"] is not None and stats["stable_step"] is not None:
        recovery_s = (
            stats["stable_step"] - stats["touchdown_step"]
        ) * env.unwrapped.step_dt
        stats["stable_recovery_margin_ms"] = (
            recovery_timeout_s - recovery_s
        ) * 1000.0

    env.close()
    candidates = sorted(recording_dir.glob("*.mp4"))
    if not candidates:
        raise RuntimeError(f"video recorder produced no mp4 in {recording_dir}")
    shutil.move(str(candidates[-1]), str(args.output))
    try:
        recording_dir.rmdir()
    except OSError:
        pass
    args.metrics.write_text(json.dumps(stats, indent=2) + "\n")
    if args.trace is not None:
        args.trace.write_text(json.dumps(trace, indent=2) + "\n")

    if args.slow_output is not None:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(args.output),
                "-vf", f"setpts={args.slow_factor}*PTS,fps={render_fps}",
                "-an", str(args.slow_output),
            ],
            check=True,
        )
    if args.slow_foot_output is not None:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(args.output),
                "-vf",
                (
                    "crop=640:360:320:360,scale=1280:720:flags=lanczos,"
                    f"setpts={args.slow_factor}*PTS,fps={render_fps}"
                ),
                "-an", str(args.slow_foot_output),
            ],
            check=True,
        )
    print(json.dumps(stats, indent=2))
    print(f"video={args.output.resolve()}")
    print(f"onnx={args.onnx.resolve()}")
    print(f"metrics={args.metrics.resolve()}")
    if args.trace is not None:
        print(f"trace={args.trace.resolve()}")
    if args.slow_output is not None:
        print(f"slow_video={args.slow_output.resolve()}")
    if args.slow_foot_output is not None:
        print(f"slow_foot_video={args.slow_foot_output.resolve()}")


if __name__ == "__main__":
    main()
