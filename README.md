# Unitree G1 reinforcement learning workspace

Workspace repository for Unitree G1 29-DOF locomotion experiments. The tracked top level contains project notes, Isaac Lab/RSL-RL training output, CMake build output, and Git submodule references for the simulation, training, and deployment code.

## Checkout requirement

The main repository references these components as Git submodules:

- `IsaacLab`
- `holosoma`
- `robot_lab`
- `rsl_rl`
- `unitree_mujoco`
- `unitree_rl_gym`
- `unitree_rl_lab`

The current repository tree does not include a `.gitmodules` file or submodule URLs. A normal clone therefore cannot populate these directories. Obtain the intended component repositories and revisions from the workspace owner before attempting setup, source changes, training, or simulation. Do not replace a gitlink with copied source without explicit approval.

## What is available here

- `Useme.md` — project-specific quick-start notes, commands, and troubleshooting.
- `G1_29dof_Velocity.md` — explanation of the G1 velocity-task policy and reward configuration.
- `Noise.md` — explanation of task randomization and the ArmHold variant.
- `logs/` — committed RSL-RL run artifacts and checkpoints.
- `outputs/` — committed Hydra output artifacts.
- `build/` — committed CMake build metadata and binaries.
- `.vscode/` — committed editor settings.

The generated artifacts above are historical workspace state, not a reproducible dependency installation or a substitute for the missing component source trees.

## Intended workflow

The existing notes describe an Isaac Lab/Isaac Sim and RSL-RL workflow on Linux with Conda. Once the required component repositories are available at the referenced paths, start with the documented commands in `Useme.md` rather than assuming a generic Python package install will work:

```bash
conda env create -f IsaacLab/environment.yml -n unitree_rl
conda activate unitree_rl
cd unitree_rl_lab
./unitree_rl_lab.sh -l
```

The documented velocity-training entry point is:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 4
```

This launches GPU-accelerated simulation and training. It is not a smoke test. Confirm compatible Isaac Sim, Isaac Lab, CUDA, and hardware versions first. For ArmHold and MuJoCo commands, use the existing, more detailed examples in `Useme.md` only after the component source and dependencies are available.

## Safe validation in this checkout

No top-level source package, test suite, dependency manifest, or executable entry point is present. Do not run training, play, CMake rebuilds, or inference from this incomplete checkout. Safe checks are limited to repository integrity and documentation review:

```bash
git status --short --branch
git fsck --no-dangling
git ls-tree HEAD
```

## Artifact and secret hygiene

- Do not add model checkpoints (`*.pt`), TensorBoard events, training logs, Hydra outputs, build directories, binaries, datasets, recordings, `.env` files, credentials, API tokens, private keys, or IDE state.
- The repository currently tracks some generated artifacts despite `.gitignore`; do not modify or restage them as part of routine documentation or source work.
- Keep new experiment outputs outside the repository or in ignored paths. Verify `git status --short` and `git diff --cached` before each commit.
- Training logs and checkpoints can disclose local paths, experiment metadata, and learned model parameters. Handle and share them according to project policy.

## Project notes

Read `Useme.md` before operating the workspace. `G1_29dof_Velocity.md` and `Noise.md` describe configuration behavior but do not replace the missing source code or upstream installation documentation.
