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
| Domain randomisation | manual | conveyor pose, cameras, object shapes, textures, clutter, table height, robot start, sensor noise, colours/sizes/lighting, friction, mass, belt speed, speed jumps, distractor, occlusion |

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
* **Entities and tasks** (`sim/entities.py`, `tasks/`, `lang/`): a pool of 8 objects is activated per episode
  (the rest are parked, collision-free). Each object has a *category* (cube, block, cylinder, slab, disc), colour,
  size and a location (conveyor or table). A **task** is an ordered list of steps "put A onto B" plus generated
  language; success is computed from generic predicates (`on`, `disturbed`, `support_height`), not hardcoded:

  | task kind | what happens |
  |---|---|
  | `stack` | pick a table object, put it on a belt object |
  | `belt_to_table` | pick a *moving* belt object, put it on a table object |
  | `belt_to_belt` | pick one belt object, put it on another |
  | `tower` | two sequential placements: A onto B (belt), then C on top of A |

  `EnvConfig(task="random")` mixes all kinds; `n_distractors` / `n_table_distractors` add bystander objects (they must
  not be knocked over: `require_unmoved`). Layouts are rejected unless every object is reachable in time for the belt
  speed and every referenced object has a unique description.
* **Language**: instructions come from a referring-expression generator that searches for a minimal *unique*
  description from category, colour, relative size ("the larger cylinder"), location and ordering ("the second cube
  from the conveyor", "the leftmost disc on the table"). Colour is only used when needed, so it cannot be a shortcut;
  the pick object is "from" a place and the support is "on" a place, followed by "onto" / "on top of". Examples:
  "put the cylinder from the conveyor onto the cylinder on the table"; "move the red cube on top of the yellow object
  on the conveyor. After that, put the cylinder on top of it". Every step also has a subgoal sentence (logged in the
  demos for hierarchical training). The policy tokenizer hashes words, so any phrasing is representable.
* **Observation**: `front` / `wrist` / `overhead` RGB, 22-D proprio (q, dq, TCP pose, commanded pose, vacuum
  cmd/held), instruction string. **Privileged** state is only in `info["priv"]`.
* **Action**: `[vx, vy, vz, wz, vacuum]` — normalised Cartesian *velocities* (1 = 0.3 m/s / 2 rad/s) integrated into a
  commanded pose and resolved by damped-least-squares IK. Bounded by construction and independent of the control rate
  (`EnvConfig.control_hz`, default 20 Hz; the expert succeeds at 10, 20 and 50 Hz).
* **Latency**: each episode samples a command latency (0-80 ms) and a camera latency (0-100 ms) scaled by `dr_scale`
  (`randomize_latency`); the expert plans from the pose that in-flight commands will produce and leads the moving
  target by the delay. `python -m vla_mujoco.sweep_rate` reports expert success vs rate and latency.
* **Chunked inference**: `PolicyRunner` updates the belief/recurrent state every control step but takes a new action
  chunk only every `--exec-horizon` steps (default 4), i.e. 5 inferences/s at 20 Hz - this is the budget a
  multi-step flow-matching head needs.
* **Success**: block supported by the target box, XY offset <= 15 mm, vacuum released, stable for 0.5 s while the
  belt keeps moving.
* **Domain randomisation** (`sim/randomize.py`, resampled every `reset`, seeded). Scale all ranges with
  `EnvConfig.dr_scale` (`--dr-scale 1.5` for out-of-distribution tests); switch groups off individually
  (`randomize_conveyor|camera|appearance|shape|texture|clutter|table|robot_start|image`) or all at once with
  `EnvConfig.dr_flags(False)` / `--no-dr`:

  | group | what varies |
  |---|---|
  | conveyor | belt centre x +-3 cm, yaw +-0.1 rad (expert follows the true belt direction and aligns the held object with the target's heading), belt/rail colours, stripe density |
  | camera pose | front/overhead/wrist position, orientation, field of view |
  | shape | the stacked object, the target and the distractor are each a cube, a cuboid (random x/y/z extents) or an upright cylinder; inertia and collision bounds are recomputed. All have flat tops; the 4-cup VGC10 seal requires every cup centre on the top face, so sizes are bounded below by the cup layout (cylinder R >= 5 cm, cuboid half-extents >= 4.5 cm) |
  | appearance | object colours jittered around their named colour (instructions stay valid), size jitter, table/floor/belt/robot tints, light positions/intensity, headlight |
  | texture | per-object / table / floor texture from a pool (checker, gradient, speckle, stripes) or plain |
  | clutter | 0-6 visual-only distractor shapes (box/cylinder/sphere) beside or behind the belt, never in the work area |
  | table | pick-up surface height 0 to -4 cm (the robot stands on a pedestal) |
  | robot start | initial TCP pose +-7 cm / +-0.4 rad around home (via IK) |
  | latency | command latency 0-80 ms, camera latency 0-100 ms |
  | image | per-episode sensor model: pixel noise, blur, exposure/bias (applied to the observation images) |
  | physics | friction, block mass, belt speed, speed jumps |

  Not randomised: arm/gripper dynamics, camera latency/lens distortion, object materials beyond friction, 3-D
  background geometry, non-flat-topped objects (suction would need a different grasp model).
* **Hard modes** (`EnvConfig`): `belt_speed_range`, `speed_change` (surprise speed jump), `occlusion_s` (a panel
  hides the target from the `front` camera — the wrist camera still sees it), `n_distractors`.

## Expert and data
The expert is a generic plan executor: for each step it runs approach, grasp, lift, transfer, place, release, retreat with
intercepts and closed-loop tracking for *moving* pick and place targets, hover height from the tallest object, the place
height from the support query, and re-grasp on a dropped or unsealed object.

Measured expert success (this repo; ad-hoc sweeps, seeds not cherry-picked; the learned policy is *not* evaluated here):

Multi-object tasks (scripted expert, full randomisation incl. language):

| setting | success |
|---|---|
| all four task kinds mixed (`task="random"`), 1 belt bystander | 100/100 |
| crowded: 3 belt + 2 table bystanders, randomisation x1.5 | 79/80 |
| fast belt 0.07-0.12 m/s + speed jump + occlusion + 2 bystanders | 58/60 |

Control rate x fixed command latency (`python -m vla_mujoco.sweep_rate`, 30 `stack` episodes each, expert success):

| rate | 0 ms | 80 ms | 160 ms | 240 ms |
|---|---|---|---|---|
| 10 Hz | 30/30 | 30/30 | 30/30 | 30/30 |
| 20 Hz | 30/30 | 30/30 | 30/30 | 30/30 |
| 50 Hz | 30/30 | 30/30 | 30/30 | 30/30 |

`data/collect.py` stores only successful episodes (npz: images, proprio, actions + privileged labels for auxiliary
losses). `--noise` adds DART-style action noise to widen state coverage.

## Policy (`vla/policy.py`, `vla/text.py`, `vla/train.py`, `vla/evaluate.py`)
```
cameras -> CNN -> patch tokens --+
text -> backend/encoder ---------+--> fusion transformer (query token) -> per-step feature
proprio -------------------------+                                         |
                                       memory: mlp | stack | gru  ->  belief z_t
                                                                            |
              heads: action chunk (regress | flow matching), vacuum (Bernoulli), aux (target-relative position,
                     target speed, phase, step)
```
* **Language grounding**: text tokens, image patch tokens and proprioception attend to each other in a small
  transformer, so "the second cube from the conveyor" can select an object in the image (late concatenation of pooled
  features cannot).
* **Text encoders** (`--text-encoder`):
  `bow` hashed words embedded from scratch (baseline) | `scratch` the same ids contextualised by a small trainable
  transformer | `pretrained` token embeddings from a **frozen** Hugging Face encoder
  (`--text-name`, default `sentence-transformers/all-MiniLM-L6-v2`) with a trainable adapter. Embeddings are computed
  once per unique instruction and cached, so the language model costs nothing per step, and it is not stored in the
  checkpoint (it is re-loaded for evaluation).
* **Action head** (`--head`): `regress` (L1) or `flow` (conditional flow matching on the action chunk, Euler sampling,
  `--flow-steps`). The vacuum command is a separate Bernoulli head. Chunked execution: the belief updates every step,
  a new chunk is drawn every `--exec-horizon` steps.
* **Memory** (`--arch`): `mlp` | `stack` | `gru`, trained with truncated BPTT and a burn-in (`--burn-in`).
* **Trainer**: episode-level train/val split, EMA weights, warm-up + cosine LR, bf16 (`--amp`), grad clipping, resumable
  (`--resume`), background batch prefetching, `log.csv`, `last.pt` / `best.pt` / `policy.pt` (final EMA).
* **Language probe**: every validation reports the action error with the correct instruction and with an instruction from
  another episode; `lang_gap = wrong - correct` is how much the policy actually uses the text (0 means it ignores it).
* The auxiliary heads are training signals only; inference never sees privileged state.

```bash
python -m vla_mujoco.data.collect --out data/mj_lang --episodes 300 --img-size 64 --task random --distractors 2
python -m vla_mujoco.vla.train --data data/mj_lang --out runs/gru_pre --arch gru --head flow \
    --text-encoder pretrained --steps 20000 --amp          # on a GPU
python -m vla_mujoco.vla.evaluate --policy runs/gru_pre/best.pt --episodes 50 --task random
```

## Known limitations / next steps
* Suction is a constraint model; no vacuum leakage, payload limits or cup compliance.
* Single scripted skill; no re-grasp or failure recovery in the expert.
* Arm meshes are the UFACTORY collision STLs (visual DAEs not vendored); IK uses a fixed tool-down orientation.
* The learned-policy numbers in the PR description come from a short CPU run and are not a benchmark.
* Natural next steps: sim-to-real camera calibration, ROS 2 command adapter around the same action interface,
  LeRobot/RLDS export, larger/pretrained backbones, DAgger with the expert.
