# Microduck Maximum-Height Jump

This repository publishes the deployable ONNX export and reproducibility
evidence for a one-shot Microduck jump trained with PPO in
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl).

The policy is **simulation-only and hardware-unverified**. The name describes
its height-optimizing training objective; it is not a claim that 31.67 mm is a
proven physical maximum.

[Watch the four-times slow-motion jump and standing handoff](media/preview-4x.mp4).

## Artifacts

- `policy/max_height_jump.onnx` — primary baked-normalizer jump policy.
- `checkpoint/model_34995.pt` — source RSL-RL checkpoint. PyTorch checkpoints
  use pickle; load it only in a trusted environment.
- `handoff/standing_policy.onnx` — companion standing policy used by the
  demonstrated post-jump reset.
- `handoff/manifest.json` — exact command and smoothstep handoff contract.
- `evaluation/` — deterministic standalone and handoff metrics plus the
  handoff trace.
- `training/` — captured agent/environment configuration, tracked working-tree
  patch, and the untracked source files present during training.
- `CHECKSUMS.sha256` — hashes for every published binary and evidence file.

## Runtime contract

Both ONNX models use the shared Microduck runtime contract:

- input: float32 `obs`, shape `[1, 61]`;
- output: float32 `actions`, shape `[1, 14]`;
- control rate: 50 Hz;
- action scale: 1.0;
- policy observations: 48 proprioception values followed by
  `twist[3]`, `head_pose[4]`, and `body_pose[6]`;
- action order: five left-leg joints, four neck/head joints, then five
  right-leg joints.

The empirical observation normalizer is embedded in each ONNX graph. Do not
apply a second normalizer or an unmatched action low-pass filter.

### One-shot command

The jump controller reuses the twist-x command slot as a binary request:

- `twist = [1, 0, 0]` requests the launch;
- `twist = [0, 0, 0]` requests settling after touchdown;
- head-pose and body-pose command slots remain zero.

Training cleared the flag on touchdown, with a 0.75 s launch-window backstop.
A deployment must provide the same stateful one-shot behavior; holding the
flag at 1 indefinitely is outside the trained contract.

## Important handoff limitation

`max_height_jump.onnx` contains only the jump controller. In direct ONNX
Runtime evaluation it achieved a qualified takeoff and crossed the stable
landing threshold with 40 ms of recovery margin, but it **did not** meet the
durable final-standing criterion: its head remained rotated and its feet
finished only 37.26 mm apart.

The quiet reset shown in the preview runs both ONNX models and blends their
14 actions with smoothstep interpolation. The blend starts 0.22 s after the
jump trigger, lasts 0.14 s, and then retains the standing controller. The
exact formula and hashes are in [`handoff/manifest.json`](handoff/manifest.json).
This orchestration is not embedded in the primary ONNX and is not supplied by
a plain `robotd.toml` policy path.

## Deterministic simulation evidence

These are direct ONNX Runtime rollouts in one seeded CPU MuJoCo environment,
not a randomized robustness battery and not physical-robot evidence.

| Metric | Jump ONNX alone | Jump-to-standing handoff |
| --- | ---: | ---: |
| Launch velocity | 0.628 m/s | 0.628 m/s |
| Visible whole-body rise | 31.67 mm | 31.67 mm |
| Bilateral sole clearance | 31.67 mm | 31.67 mm |
| Airtime | 140 ms | 140 ms |
| Stable landing | yes | yes |
| Durable landing | **no** | **yes** |
| Failed recovery | no | no |
| Stable-recovery margin | 40 ms | 100 ms |
| Settle score | 0.778 | 0.860 |
| Non-foot body contact | no | no |
| Final tilt | 3.74 degrees | 2.87 degrees |
| Maximum final head-joint error | 44.02 degrees | 7.27 degrees |
| Final foot spacing | 37.26 mm | 81.95 mm |

The simulation used the standard all-collisions robot model, BAM M6 voltage
actuation, voltage sag, actuator delay, and domain randomization. It did not
use the backlash model. Real backlash, battery state, timing jitter, floor
contact, and state estimation can invalidate this result. Validate in a safe
support rig before attempting any hardware jump.

## Reconstructing the training tree

The captured base revision is
[`d424a0c899f6b33cbd3daeb279913134349c0b63`](https://github.com/pollen-robotics/microduck_rl/commit/d424a0c899f6b33cbd3daeb279913134349c0b63).
See [`training/README.md`](training/README.md) for the exact reconstruction
steps and limitations.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Microduck and the base training project
are maintained by Pollen Robotics; this community policy is not upstream- or
hardware-verified.
