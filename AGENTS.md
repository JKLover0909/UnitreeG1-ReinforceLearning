# AGENTS.md

## Purpose

Maintain the top-level Unitree G1 workspace safely. This checkout exposes notes and historical artifacts, but not the source code behind its Git links.

## Hard boundaries

- `IsaacLab`, `holosoma`, `robot_lab`, `rsl_rl`, `unitree_mujoco`, `unitree_rl_gym`, and `unitree_rl_lab` are gitlinks, not checked-out source directories.
- `.gitmodules` is absent. Do not guess submodule URLs, run recursive initialization, populate directories manually, or replace gitlinks.
- Do not edit, delete, regenerate, or stage `build/`, `logs/`, `outputs/`, `.vscode/`, checkpoints, binaries, or their contents.
- Do not infer unverified APIs, package versions, test commands, or runtime behavior from descriptive notes alone.

## Before an edit

```bash
git status --short --branch
git ls-tree HEAD
git diff --check
```

Read the applicable top-level note fully: `Useme.md`, `G1_29dof_Velocity.md`, or `Noise.md`. Limit edits to files that are present and within the requested scope.

## Runtime restrictions

Do not launch Isaac Sim, Isaac Lab, RSL-RL training, play/inference, MuJoCo simulation, robot control, CMake configuration/builds, checkpoint conversion, or downloads as a default check. These workflows require missing component source, a mapped dependency environment, GPU resources, and potentially physical-hardware safeguards.

When source access is explicitly restored, verify the supplied remotes and pinned revisions before coding. Use the source repository's own documented setup and test commands rather than these historical workspace artifacts.

## Validation and review

The valid lightweight checks for this checkout are Git integrity and documentation diff review:

```bash
git fsck --no-dangling
git ls-tree HEAD
git diff --check
git diff --cached --check
git diff --cached -- README.md CLAUDE.md AGENTS.md
```

State explicitly when simulator, training, or integration validation is skipped because the checkout lacks `.gitmodules` mapping and component source.

## Commit safety

- Stage only named, intended files.
- Never commit secrets, `.env` files, credentials, private keys, PII, datasets, recordings, model weights, TensorBoard events, Hydra output, logs, build products, binaries, caches, or editor state.
- Confirm the staged file list and diff before commit. A documentation-only task must contain only documentation files.
