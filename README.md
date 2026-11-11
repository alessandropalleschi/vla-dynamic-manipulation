# VLA for Dynamic Robotic Manipulation

A simulation framework for **Vision-Language-Action (VLA) learning** on *dynamic stacking*: a UF850 arm with an
OnRobot VGC10 vacuum gripper picks a block and stacks it onto a box that is **moving on a conveyor**, with
language-specified targets, distractors, speed changes and visual occlusion.

![status](https://img.shields.io/badge/sim-MuJoCo-blue) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

| Stack | Location | Status |
|---|---|---|
| **MuJoCo** (fast, deterministic, no ROS) | [`vla_mujoco/`](vla_mujoco/) | active — environment, scripted expert, data pipeline, VLA baseline |
| Gazebo Harmonic + ROS 2 + MoveIt 2 | `scripts/`, `docker/`, `simulation/`, `compose*.yaml` | legacy, kept working |

## Quick start (MuJoCo)

> **Git LFS:** meshes (`*.stl`, `*.dae`, `*.obj`, ...) are stored with [Git LFS](https://git-lfs.com). Install it before
> cloning (`git lfs install`), or run `git lfs pull` afterwards; without it the meshes are tiny pointer files and the
> simulator stops with an error that says so.

**VS Code:** open the folder, run *Dev Containers: Reopen in Container* and pick a configuration:

| Configuration | Use when | What you get |
|---|---|---|
| `vla-mujoco (GPU, NVIDIA)` | the host has an NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) | CUDA torch for training, MuJoCo cameras rendered by the GPU through EGL |
| `vla-mujoco (CPU only)` | no GPU / CI-like setups | CPU torch, software rendering |

Both install Python 3.12, all dependencies and the recommended extensions, then run the tests and `make gpu-check`.
Without VS Code: `docker compose -f compose.mujoco.yaml run --rm mujoco` gives a GPU shell.

**Local:**
```bash
python -m venv .venv && source .venv/bin/activate
make install                  # = pip install -e ".[train,dev]"
export MUJOCO_GL=egl          # headless GPU/software rendering (glfw if you have a display)
make test                     # simulator tests, ~3 s
make demo                     # expert episode -> demo.gif
make collect                  # 300 mixed-task expert demos -> data/mj_demos
make gpu-check                # is CUDA + GPU rendering working?
make train                    # GRU belief + pretrained text encoder + flow-matching head -> runs/gru_flow
```
A GL-less container needs `apt-get install libegl1` (Mesa software rendering is then used).

## What's in the MuJoCo stack

```
vla_mujoco/
  sim/      scene.py (MJCF generator), env.py (Gymnasium-style DynamicStackEnv), assets/ (UF850 meshes)
  entities.py, randomize.py   (inside sim/)
  tasks/    goal predicates, task steps, scene/task generator
  lang/     referring-expression and instruction generation
  expert.py generic privileged expert: pick/place any object (static or moving), multi-step plans
  data/     collect.py   parallel demo collection (images, proprio, actions, privileged labels)
  vla/      policy.py (CNN + text + proprio -> action chunks; mlp | stack | gru memory ablation),
            train.py, evaluate.py
  viz.py    render an episode to GIF
tests/      simulator + expert tests (no GPU, no rendering)
docs/       MUJOCO.md (design, limitations), next_steps.md (research plan), legacy/ (Gazebo-era notes)
```

* **Tasks** — four task kinds (`stack`, `belt_to_table`, `belt_to_belt`, 2-step `tower`) over a pool of objects
  (cube, block, cylinder, slab, disc) on both the table and the moving belt, with bystanders that must not be disturbed.
  Instructions are generated from the scene (category, colour, size, location, order on the belt), e.g.
  *"put the cylinder from the conveyor onto the cylinder on the table"*.
* **Observations** — `front` / `wrist` / `overhead` RGB, 22-D proprioception, language instruction. Privileged state is
  confined to `info["priv"]`, so deployed policies never see it.
* **Actions** — normalised Cartesian velocities `[vx, vy, vz, wz, vacuum]` resolved by IK, independent of the control
  rate; command/camera latency is randomised; policies replan action chunks every few steps (`--exec-horizon`).
* **Domain randomisation** — conveyor position/yaw, camera poses, **stacked/target object shapes** (cube, cuboid,
  cylinder), textures, colours, lighting, background clutter, table height, robot start pose and camera sensor
  noise/blur are resampled every episode (`--dr-scale`, `--no-dr`); see `docs/MUJOCO.md` for the full table.
* **Hard modes** — belt speed range, surprise speed jumps, an occluder panel hiding the target from the front camera,
  distractor boxes (the instruction names colours, so language must be grounded).
* **Expert** — solves 100/100 mixed multi-object tasks with full randomisation, 79/80 in crowded scenes at 1.5× randomisation, and 58/60
  with fast belts, speed jumps and occlusion (see `docs/MUJOCO.md`).
* **Policy** — image patch tokens, text tokens (hashed / trainable / frozen pretrained encoder) and proprioception are
  fused by a transformer, then a memory core (`mlp | stack | gru`) and an action head (`regress | flow`). The trainer has
  a validation split, EMA, resumable checkpoints and a *language probe* that measures how much the policy uses the text.
* **Console scripts** after install: `vla-demo`, `vla-collect`, `vla-train`, `vla-eval`.

Read [`docs/MUJOCO.md`](docs/MUJOCO.md) for the gripper model, success definition, limitations and next steps.
The learned-policy baseline is a scaffold (no pretrained weights); no benchmark numbers are claimed for it yet.

## Research directions

Dynamic manipulation and moving-target interception · vision-based tracking · learning from demonstrations ·
belief-state estimation under occlusion · VLA policies. The planned memory comparison (single-frame vs window vs
recurrent vs RSSM) is described in [`docs/next_steps.md`](docs/next_steps.md); the multi-object / multi-step task and
belief-conditioned flow-matching policy design is in [`docs/TASK_AND_POLICY_DESIGN.md`](docs/TASK_AND_POLICY_DESIGN.md).

## Development

```bash
make hooks     # once per clone (make install does it too): git pre-commit hooks
make lint      # ruff check + format check (legacy scripts/ is excluded)
make format
make test
```
The **pre-commit hooks** reject files over 1 MB and any checkpoint, dataset, recording or `runs*/` / `data/` output
(`.pre-commit-config.yaml`; meshes are exempt because they are stored in Git LFS, see `.gitattributes`), so training artifacts cannot be committed by accident.
CI (`.github/workflows/ci.yml`) runs the same hooks and the tests on every push. VS Code tasks/launch configs for demo, collection,
training and evaluation are in `.vscode/`.

## Legacy Gazebo / ROS 2 stack

ROS 2 Jazzy + Gazebo Harmonic + MoveIt 2 in Docker (requires an NVIDIA GPU host):

```bash
bash scripts/run_uf850_moving_stack_trial.sh      # moving-target stacking trial
bash scripts/record_and_run_uf850_v9.sh           # synchronized RGB demonstration
```
Its scripts hardcode `/workspace/...` container paths and therefore stay under `scripts/`
(see [`scripts/README.md`](scripts/README.md)). Historical version notes are in [`docs/legacy/`](docs/legacy/).

## Licenses and third-party assets

UF850 meshes/kinematics: UFACTORY `xarm_ros2` (BSD-3, `vla_mujoco/sim/assets/uf850/LICENSE_xarm_ros2.txt`).
VGC10 meshes: `UOsaka-Harada-Laboratory/onrobot` (MIT, `external/onrobot_ros_assets/`).
