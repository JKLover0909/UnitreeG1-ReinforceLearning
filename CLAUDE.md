# CLAUDE.md

## Checkout boundary

This repository is a workspace shell, not a self-contained source checkout. `IsaacLab`, `holosoma`, `robot_lab`, `rsl_rl`, `unitree_mujoco`, `unitree_rl_gym`, and `unitree_rl_lab` are Git links in the main tree. The committed tree has no `.gitmodules` file or URL mapping, so those source directories cannot be initialized from this checkout.

Do not invent component remotes, create replacement source trees, convert git links to normal directories, or make claims about source behavior that cannot be verified from the current tree. Ask for the intended component repository URLs and revisions before source-level work.

## Tracked workspace state

- `Useme.md`, `G1_29dof_Velocity.md`, and `Noise.md` provide project notes and examples.
- `build/`, `logs/`, `outputs/`, and `.vscode/` are already tracked historical artifacts.
- Git ignores identify common generated outputs, but existing tracked artifacts remain tracked.

Never alter, remove, reformat, regenerate, or restage tracked build, log, output, checkpoint, or gitlink entries for a routine documentation task.

## Agent workflow

1. Inspect `git status --short`, `git ls-tree HEAD`, the relevant top-level notes, and gitlink state before making a change.
2. Work only in files available in this checkout. Do not run commands in the empty gitlink directories.
3. Treat all commands in `Useme.md` as environment-specific operational guidance. They require a completed component checkout and compatible Isaac Sim/Isaac Lab/CUDA setup.
4. Do not run training, play, MuJoCo simulation, deployment/control programs, CMake rebuilds, checkpoint loading, model conversion, or downloads as a routine validation step.
5. If the user supplies component remotes, initialize them only with explicit scope and validate their revisions before changing source.

## Validation

No usable top-level source package, test target, dependency manifest, or executable entry point is present. Use non-invasive checks only:

```bash
git status --short --branch
git fsck --no-dangling
git ls-tree HEAD
git diff --check
```

For documentation-only edits, review the staged diff:

```bash
git diff --cached --check
git diff --cached -- README.md CLAUDE.md AGENTS.md
```

Report that simulator and training validation was not run because the required gitlink source and environment mapping are unavailable.

## Data and commit hygiene

- Never commit credentials, `.env` files, tokens, private keys, PII, raw datasets, recordings, checkpoints, TensorBoard events, Hydra outputs, model exports, binaries, caches, virtual environments, build products, or IDE state.
- Do not stage existing changes in `build/`, `logs/`, `outputs/`, `.vscode/`, or any gitlink.
- Stage only explicit intended files. Review `git status --short`, `git diff --check`, and `git diff --cached` before commit.
