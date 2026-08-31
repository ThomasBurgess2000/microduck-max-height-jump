# Captured training state

The final run started from `pollen-robotics/microduck_rl` commit
`d424a0c899f6b33cbd3daeb279913134349c0b63` with a dirty working tree.
This artifact captures the tracked-file delta in `working-tree.diff`; relevant
untracked sources are preserved under `untracked/` with their intended
repository-relative paths. The included evaluation script has a post-training
correction that uses the full 175-step episode horizon by default.

To reconstruct the recorded tree in a disposable clone:

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
git checkout d424a0c899f6b33cbd3daeb279913134349c0b63
git apply /path/to/training/working-tree.diff
cp -R /path/to/training/untracked/. .
```

Then use `agent.yaml` and `env.yaml` as the resolved run configurations. The
policy source checkpoint is `../checkpoint/model_34995.pt`.

The reconstruction sequence above was checked successfully against the stated
base revision before this artifact was published.

The captured patch also contains local W&B and utility changes that were in
the working tree when the run started. They are retained for provenance, not
presented as the minimal jump implementation. A future upstream source commit
should replace this snapshot before the policy is called reproducible from a
normal branch.
