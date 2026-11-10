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
| Domain randomisation | manual | colors, friction, mass, belt speed, speed jumps, distractor, occlusion |

## Quick start
```bash
pip install -r requirements-mujoco.txt
export MUJOCO_GL=egl            # headless; use `glfw` with a display, `osmesa` if no EGL
pytest tests/test_mujoco_env.py
python -m vla_mujoco.render_demo --seed 3 --out demo.gif      # watch the expert
python -m vla_mujoco.collect --out data/mj_demos --episodes 200 --workers 4
python -m vla_mujoco.train --data data/mj_demos --arch gru --steps 20000 --out runs/gru
python -m vla_mujoco.evaluate --policy runs/gru/policy.pt --episodes 50
python -m vla_mujoco.evaluate --expert --episodes 50 --speed-range 0.08 0.14
```
(For a GL-less container: `apt-get install libegl1`; EGL then falls back to llvmpipe.)

## Environment (`vla_mujoco/env.py`)
* **Robot**: UF850 kinematics, joint limits and (collision) meshes from UFACTORY `xarm_ros2` (BSD-3, license
  in `assets/uf850/`). Arm is gravity-compensated with position actuators; masses are approximate.
* **Tool**: VGC10-sized body, 4 cups, TCP at the cup face. Vacuum is modelled as a weld constraint that engages
  only after the cups have sealed on a box top face (gap < 8 mm, lateral < 15 mm, tilt < 14 deg) for 0.35 s and
  releases 0.2 s after the command drops. This is a *model*, not a measured VGC10 — calibrate before drawing
  conclusions about real hardware.
* **Conveyor**: a belt slab on a velocity-driven slide joint, so boxes are carried by real friction, and can slip.
* **Task**: pick the *source block* from the table, stack it on the *target box* (one of two boxes on the belt;
  the language instruction names the colours, so the distractor requires language grounding).
* **Observation**: `front` / `wrist` / `overhead` RGB, 22-D proprio (q, dq, TCP pose, commanded pose, vacuum
  cmd/held), instruction string. **Privileged** state is only in `info["priv"]`.
* **Action**: `[dx, dy, dz, dyaw, vacuum]` — bounded Cartesian deltas (15 mm / 0.1 rad per 20 Hz step) resolved by
  damped-least-squares IK, so the policy output is bounded by construction.
* **Success**: block supported by the target box, XY offset <= 15 mm, vacuum released, stable for 0.5 s while the
  belt keeps moving.
* **Hard modes** (`EnvConfig`): `belt_speed_range`, `speed_change` (surprise speed jump), `occlusion_s` (a panel
  hides the target from the `front` camera — the wrist camera still sees it), `n_distractors`.

## Expert and data
`StackExpert` (privileged) does approach → grasp → lift → constant-velocity intercept → closed-loop track-and-descend
→ timed release. Current measured success (this repo, `tests/` + ad-hoc sweeps over 30-60 seeds):

| setting | expert success |
|---|---|
| default belt 0.02-0.07 m/s, distractor | 60/60 |
| speed jump + occlusion | 40/40 |
| fast belt 0.08-0.14 m/s | 39/40 |

`collect.py` stores only successful episodes (npz: images, proprio, actions + privileged labels for auxiliary
losses). `--noise` adds DART-style action noise to widen state coverage.

## Policy (`policy.py`, `train.py`, `evaluate.py`)
Compact baseline: CNN (spatial-softmax) per camera + word-level text encoder + proprio MLP -> action **chunks**
(8 steps), with an auxiliary head that regresses the target's position relative to the tool and belt velocity
(a belief-style training signal; unused at inference). `--arch {mlp,stack,gru}` gives the memory ablation from
`docs/next_steps.md` (no memory / frame window / recurrent). This is a *scaffold*: no pretrained weights are
downloaded. To scale up, swap `ImageEncoder`/`TextEncoder` for a frozen ViT + language model and the head for
diffusion/flow matching.

## Known limitations / next steps
* Suction is a constraint model; no vacuum leakage, payload limits or cup compliance.
* Single scripted skill; no re-grasp or failure recovery in the expert.
* Meshes are the collision STLs (visual DAEs are not vendored); IK uses a fixed tool-down orientation.
* The learned-policy numbers in the PR description come from a short CPU run and are not a benchmark.
* Natural next steps: sim-to-real camera calibration, ROS 2 command adapter around the same action interface,
  LeRobot/RLDS export, larger/pretrained backbones, DAgger with the expert.
