# MuJoCo simulation stack (`vla_mujoco/`)

A self-contained replacement for the Gazebo/ROS 2 stack for VLA research on **dynamic stacking**:
a UF850 with a VGC10-style vacuum tool picks a block and stacks it on a box riding a moving conveyor.
No ROS, no Docker, no GPU needed for simulation (headless EGL rendering works in software).

## Why MuJoCo here
| | Gazebo stack (legacy `scripts/`) | `vla_mujoco` |
|---|---|---|
| Speed | wall-clock bound, ROS bridges | ~10-20x faster than real time incl. 2 cameras; physics alone >100x |
| Determinism | timing-dependent (bridges, `gz topic`) | seeded `reset(seed)`, fixed step |
| Contact quality | ODE/DART, soft | MuJoCo implicit contacts, tuned friction |
| Suction | pre-engaged / fixture | seal check + 0.35 s grip / 0.2 s release delay (`config/vgc10.yaml`) |
| Episode setup | edit SDF, restart sim | `env.reset(seed, options)` in-process |
| Domain randomisation | manual | conveyor pose, cameras, colours/sizes/lighting, friction, mass, belt speed, speed jumps, distractor, occlusion |

## Quick start
See the README for the dev-container flow. Manually:
```bash
pip install -e ".[train,dev]"
export MUJOCO_GL=egl            # headless; use `glfw` with a display, `osmesa` if no EGL
pytest
python -m vla_mujoco.viz --seed 3 --out demo.gif                       # watch the expert  (vla-demo)
python -m vla_mujoco.data.collect --out data/mj_demos --episodes 200 --workers 4   # (vla-collect)
python -m vla_mujoco.vla.train --data data/mj_demos --arch gru --steps 20000 --out runs/gru   # (vla-train)
python -m vla_mujoco.vla.evaluate --policy runs/gru/policy.pt --episodes 50                   # (vla-eval)
python -m vla_mujoco.vla.evaluate --expert --episodes 50 --speed-range 0.08 0.14
```
(For a GL-less container: `apt-get install libegl1`; EGL then falls back to llvmpipe.)

## GPU
Simulation physics is CPU-only; the GPU helps in two places:

* **Training / policy inference** — `vla-train` and `vla-eval` use CUDA automatically (`--device auto|cpu|cuda[:N]`);
  `--amp` enables bf16 autocast on CUDA, TF32 and cuDNN autotuning are switched on. Batches are built on the CPU and
  moved with `non_blocking` copies; for very large datasets raise `--batch`/`--seq` until the GPU is saturated.
* **Camera rendering** — with `MUJOCO_GL=egl` and an NVIDIA container started with the `graphics` capability, MuJoCo
  renders on the GPU instead of llvmpipe (`make gpu-check` prints the active `GL_RENDERER` and ms/frame; it warns when
  a software rasteriser is in use). Parallel collection (`--workers N`) opens one EGL context per worker.

Host requirement: NVIDIA driver + NVIDIA Container Toolkit (`docker run --gpus all ...`). The dev-container GPU
profile passes `--gpus=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display` and `--shm-size=8g`, and
installs the default PyPI torch wheel (CUDA build; it bundles the CUDA runtime, so only the driver is needed).
If `make gpu-check` shows `GL_RENDERER=llvmpipe` inside the GPU container, the toolkit did not inject
`libEGL_nvidia`; check that the `graphics` capability is enabled (`post-create.sh` also writes the EGL vendor file when
the library exists but the file does not). Rendering is correct either way, just slower.

## Environment (`vla_mujoco/sim/env.py`)
* **Robot**: UF850 kinematics, joint limits and (collision) meshes from UFACTORY `xarm_ros2` (BSD-3, license
  in `sim/assets/uf850/`). Arm is gravity-compensated with position actuators; masses are approximate.
* **Tool**: the same VGC10 setup as the Gazebo stack (`scripts/uf850_tool_urdf.py`): mount 15 mm from link6 flipped
  180 deg, 101x100x100 mm body proxy for collision, 4 cups of 40 mm / 44 mm pitch, TCP 0.1894 m from link6. The
  real OnRobot `base_link_4cups.stl` / `suction_cup_1cup.stl` (MIT, `external/onrobot_ros_assets`) are used as
  **visual only**, exactly as in the Gazebo stack; if they are missing a primitive fallback is built.
  Vacuum is a weld constraint that engages only after all four cup centres lie on a box top face (with margin),
  the tool is within 8 mm / 14 deg, for 0.35 s; it releases 0.2 s after the command drops. Because the cups span
  ~84 mm, blocks are 10 cm (source) and 12 cm (target) cubes. This is a *model*, not a measured VGC10, and
  `config/vgc10.yaml` (30 mm / 50 mm cups) disagrees with the URDF generator (40 mm / 44 mm); the generator
  values are used. Channels A/B are not modelled separately.
* **Conveyor**: a belt slab on a velocity-driven slide joint, so boxes are carried by real friction, and can slip.
* **Task**: pick the *source block* from the table, stack it on the *target box* (one of two boxes on the belt;
  the language instruction names the colours, so the distractor requires language grounding).
* **Observation**: `front` / `wrist` / `overhead` RGB, 22-D proprio (q, dq, TCP pose, commanded pose, vacuum
  cmd/held), instruction string. **Privileged** state is only in `info["priv"]`.
* **Action**: `[dx, dy, dz, dyaw, vacuum]` — bounded Cartesian deltas (15 mm / 0.1 rad per 20 Hz step) resolved by
  damped-least-squares IK, so the policy output is bounded by construction.
* **Success**: block supported by the target box, XY offset <= 15 mm, vacuum released, stable for 0.5 s while the
  belt keeps moving.
* **Domain randomisation** (`sim/randomize.py`, resampled each `reset`, seeded; scale all ranges with
  `EnvConfig.dr_scale`, switch groups off with `randomize_conveyor|camera|appearance=False` or `--no-dr`):
  * *Conveyor*: belt centre x +-3 cm, belt yaw +-0.1 rad (the expert follows the true belt direction and aligns the
    held block with the target's heading), belt/rail colours.
  * *Cameras*: front/overhead/wrist position, orientation and field of view.
  * *Appearance*: object colours jittered around their named colour (instructions stay valid), block/box sizes
    (+-5 mm / +-8 mm), table, floor, belt and robot tints, light positions/intensity, headlight.
  * *Physics*: friction, block mass, belt speed, speed jumps.
  Use `--dr-scale 1.5` in `vla-eval` for out-of-distribution robustness tests. Not randomised (yet): textures,
  distractor clutter, camera noise/blur (training adds photometric noise and shifts), table height, robot pose
  beyond the home configuration.
* **Hard modes** (`EnvConfig`): `belt_speed_range`, `speed_change` (surprise speed jump), `occlusion_s` (a panel
  hides the target from the `front` camera — the wrist camera still sees it), `n_distractors`.

## Expert and data
`StackExpert` (privileged) does approach → grasp → lift → constant-velocity intercept → closed-loop track-and-descend
→ timed release. Current measured success (this repo, `tests/` + ad-hoc sweeps over 40 seeds each):

| setting | expert success |
|---|---|
| default belt 0.02-0.07 m/s, distractor, full domain randomisation | 60/60 |
| same, randomisation ranges x1.6 | 60/60 |
| speed jump + occlusion | 40/40 (nominal scene) |
| fast belt 0.08-0.14 m/s + speed jump + occlusion + randomisation | 58/60 |

`data/collect.py` stores only successful episodes (npz: images, proprio, actions + privileged labels for auxiliary
losses). `--noise` adds DART-style action noise to widen state coverage.

## Policy (`vla/policy.py`, `vla/train.py`, `vla/evaluate.py`)
Compact baseline: CNN (spatial-softmax) per camera + word-level text encoder + proprio MLP -> action **chunks**
(8 steps), with an auxiliary head that regresses the target's position relative to the tool and belt velocity
(a belief-style training signal; unused at inference). `--arch {mlp,stack,gru}` gives the memory ablation from
`docs/next_steps.md` (no memory / frame window / recurrent). This is a *scaffold*: no pretrained weights are
downloaded. To scale up, swap `ImageEncoder`/`TextEncoder` for a frozen ViT + language model and the head for
diffusion/flow matching.

## Known limitations / next steps
* Suction is a constraint model; no vacuum leakage, payload limits or cup compliance.
* Single scripted skill; no re-grasp or failure recovery in the expert.
* Arm meshes are the UFACTORY collision STLs (visual DAEs not vendored); IK uses a fixed tool-down orientation.
* The learned-policy numbers in the PR description come from a short CPU run and are not a benchmark.
* Natural next steps: sim-to-real camera calibration, ROS 2 command adapter around the same action interface,
  LeRobot/RLDS export, larger/pretrained backbones, DAgger with the expert.
