# Handoff: VLA for dynamic stacking in MuJoCo

You are taking over a working repo mid-experiment. Read this fully before touching anything. Everything below is
verified against the code and runs unless it is explicitly marked **unverified**. When in doubt, re-measure.

## 0. What this project is
A UF850 arm with a 4-cup VGC10 vacuum gripper picks objects and stacks them on a moving conveyor, in MuJoCo (the legacy
Gazebo/ROS stack under `scripts/`, `docker/`, `simulation/` is kept but not developed). Goal: a Vision-Language-Action
policy (instruction + RGB + proprioception -> actions) that does dynamic stacking, later sandwich-style multi-step
tasks. Design intent is in `docs/TASK_AND_POLICY_DESIGN.md`; simulator details in `docs/MUJOCO.md`; every experiment so
far (up to the grounding fine-tune) in `docs/TRAINING_FINDINGS.md`.

## 1. Environment
* Dev container (GPU profile): RTX 4070 12 GB (shared with other processes, ~8 GB free), 12 cores, `MUJOCO_GL=egl`
  renders on the GPU (~0.2 ms/frame). Python 3.12, torch CUDA. `make install` (also installs git hooks), `make gpu-check`.
* Meshes are **Git LFS** (`*.stl` etc.); `git lfs pull` if you see "pointer files" errors.
* Pre-commit hooks block files > 1 MB that are not LFS and any checkpoint/dataset/`runs*/`/`data/` content. `data/` and
  `runs*/` are git-ignored: never commit them.
* Tests: `pytest` (~70 tests, a few minutes; some use the GPU renderer). Lint: `ruff check . && ruff format --check vla_mujoco tests`.
* Mind `ruff format`: it rewraps code, so scripted string replacement against code you just wrote often silently misses.
  After any scripted edit, grep to confirm it applied.

## 2. Repo map (what matters)
```
vla_mujoco/
  sim/        scene.py (MJCF generator + hardware options), env.py (DynamicStackEnv), entities.py, randomize.py,
              imaging.py (letterbox), assets/{uf850,adapter,oak_d_pro}/
  tasks/      predicates.py, tasks.py, generator.py (scene + task layouts, reachability-aware roles)
  lang/       refexp.py (referring expressions + instruction templates)
  expert.py   privileged scripted expert (pick/place any object, static or moving, re-grasp)
  data/       collect.py (expert demos), static_scenes.py (grounding scenes), store.py (memmap format + manifest)
  vla/        policy.py (CNN/ViT + text + proprio fusion, mlp|stack|gru, regress|flow head, grounding slots),
              text.py, train.py, pretrain_grounding.py, dagger.py, probe.py (CI language probe), decision.py,
              stats.py, evaluate.py
tests/        incl. test_hardware.py, test_dagger.py, test_data_store.py, test_decision_and_probe.py
```
Datasets are memory-mapped (`data/store.py`); `make data` rebuilds them and writes a `manifest.json` (commit + command).

## 3. State of the working tree (all on `main`, **uncommitted**)
Last commit is `93b13dd fix post create` (the user's). Uncommitted, in two groups - commit them separately:
1. **The user's own devcontainer edits** (leave semantics alone): `.devcontainer/{cpu,gpu}/devcontainer.json`
   (pip/pre-commit cache volumes merged with the Claude/Codex mounts), `.devcontainer/post-create.sh`,
   `.devcontainer/gpu/devcontainer-lock.json` (worth committing).
2. **This session's work**: data format + manifests, decision-frame weighting, CI probe, DAgger + curriculum,
   `transformers` dependency, and the new hardware (below), plus their tests/docs/Makefile.
The full test suite had **not** been confirmed green after the hardware change when this note was written (it was
running); run `pytest` first. Lint was clean.

## 4. What is built and verified
* **Expert**: solves stack / belt_to_table / belt_to_belt / tower with full randomisation, 99-100% success (60/60 in
  the latest hardware sweep).
* **Language**: referring expressions generated from the scene (category, colour, size, location, order), unique by
  construction; subgoal sentence per step.
* **Grounding pre-training** (`pretrain_grounding.py`, 50k static scenes, 128 px, 16x16 patch grid, frozen MiniLM):
  top-1 patch accuracy 0.713 with the right instruction vs 0.248 with another scene's (gap +0.46; location prior 0.019).
  With only ~300 demos the gap was zero; it is a data-scale effect, not the text encoder.
* **Policy** (GRU belief, flow head, grounding slots, pretrained text) fine-tuned on 589 demos; the probe
  (`vla.probe`, 59 held-out episodes, bootstrap CIs) shows the actions use the instruction:
  action-error gap on decision frames +0.080 [0.062, 0.098] (decision weight 1) and +0.083 [0.068, 0.099] (weight 5).
  Decision-frame weighting made no significant difference.
* **New hardware** (`EnvConfig.wrist_mount="oak"`, `overhead_view="front"`, defaults unchanged): 15 mm flange adapter
  plate (always on; fills the gap between link6's flange face at z=0 and the VGC10 at z=15 mm), one OAK-D Pro on the
  plate's inclined wing, top camera beyond the belt looking at the robot. 60/60 expert success, 0 contacts, closest
  approach 34.5 mm to robot links (link6 excluded), 72 mm bystanders, 97 mm task objects, 88 mm table/belt. Details and
  limits in `docs/MUJOCO.md` ("Hardware"). `tests/test_hardware.py` pins the geometry.

## 5. THE open problem (read this)
**The learned policy does not succeed in closed loop.** 0/30 episodes for both fine-tuned policies, with action chunks of
4 steps and with re-planning every step; picks 1/30 and 0/30. A teacher-forced probe looks good, but rollouts fail.
Evidence gathered:
* Seal-geometry diagnostic (20 episodes, easiest scene: one table object, one slow belt object, no bystanders):
  19/20 never reached valid seal geometry (all four cup centres on the top face, gap -4..8 mm, tilt < ~14 deg); the
  "reached it but too briefly" bucket (where seal feedback would help) was **empty**. So the failure is precise
  positioning (millimetres in height/tilt), not seal timing. Median closest approach to the instructed object: 5.3 cm.
* DAgger with curriculum stage 0 (`vla.dagger`, 5 rounds, ~60 relabelled episodes/round, expert shadow labels, failures
  kept): closed-loop eval 0/20 every round (picks 0-10%, noise). A diagnostic after round 2 showed median closest approach
  2.4 cm and one full success in 12, i.e. closer but not over the threshold. Stage cap reached, gate (60%) never met.
* The old wrist camera is blind at contact (object vanishes at ~3 mm behind the tool/cups), and the front camera at 128 px
  can't tell 9 mm from 3 mm. That motivated the new hardware, which was only just built.
* Literature check (web, not read in depth): depth often *hurts* precise insertion on real sensors; wrist RGB tends to
  help; VGC10 reports vacuum only as a % per channel (A/B), not per cup - so per-cup seal feedback is not realistic.

## 6. Next steps, in order
1. `pytest` + lint; commit (two groups, see 3). Co-author trailer is fine.
2. Rebuild data for the new hardware in its own directory (~10 min total, GPU rendering):
   `make data DATA=data_oak CAMS="wrist overhead" HW="--wrist-mount oak --overhead-view front"`
   (50k scenes + 600 demos at 128 px). Existing `data/` was rendered with the *old* hardware (no adapter, old camera).
3. Re-run the pipeline on it with the user-confirmed two cameras: OAK `wrist` + repositioned `overhead`.
   The camera count matches the old setup but identities and views differ; pretrain fresh rather than reuse old camera embeddings:
   `python -m vla_mujoco.vla.pretrain_grounding --data data_oak/scenes --out runs/ground_oak --text-encoder pretrained
   --patch 8 --cameras wrist overhead --steps 10000 --batch 64 --amp` (~8 min), then
   `python -m vla_mujoco.vla.train --data data_oak/mj_demos --out runs/ft_oak --arch gru --head flow --text-encoder pretrained
   --patch 8 --slots --cameras wrist overhead --init runs/ground_oak/best.pt --steps 6000 --batch 16 --seq 24
   --burn-in 6 --lr 3e-4 --amp --decision-w 1 --val-every 500` (~18 min), then the probe, then
   `python -m vla_mujoco.vla.dagger --init runs/ft_oak/policy.pt --base-data data_oak/mj_demos --data-out data_oak/dagger
   --out runs/dagger_oak --stages s0_static_pick` (rollouts take their hardware from the checkpoint's recorded dataset config).
4. Compare against the baseline above (stage-0 closed loop 0/20; median closest approach 5.3 cm -> 2.4 cm after DAgger;
   19/20 never reached seal geometry) using `$ python -m vla_mujoco.vla.probe` and the seal-geometry diagnostic idea
   (count steps where `env._seal_candidate()` is the instructed object). Report numbers with CIs; the eval sets are small.
5. If it still fails: higher-resolution wrist crop (224 px), more/longer DAgger, check the suction model and expert
   tolerances; consider the multi-step tasks only after single picks work. Update `docs/TRAINING_FINDINGS.md`, which does
   **not** yet contain the fine-tune/probe/closed-loop/DAgger results above (they exist only in this note).

## 7. Gotchas that cost time
* Tests that check *nominal* geometry must disable randomisation (`EnvConfig.dr_flags(False)`); the randomiser jitters
  cameras and the wrist camera pose on every reset. `dr_flags` must cover every `randomize_*` switch (a test enforces it).
* MuJoCo caches contact filters per body: when activating pooled objects set `body_contype/conaffinity` as well as the geom's.
* MuJoCo meshes collide as **convex hulls**; the adapter plate is therefore visual-only with a disc cylinder + a separate
  convex wedge as collision shapes. `mj_geomDistance` ignores contact filtering (useful, but parent/child contacts are
  silently excluded in simulation).
* The simulator has **no arm self-collision**; the expert's closest approach to link5/4 is 34.5 mm, a learned policy may do worse.
* `--amp` + validation once crashed (bf16 features into float32 heads, GPU only); fixed and tested with a CPU bf16 shim.
* `vla-eval` must default its hardware flags to `None` and read them from the checkpoint; defaulting to "legacy" silently
  evaluates an OAK-trained policy on the wrong camera (tested).
* Use `spawn`, not fork, for multiprocessing after torch/GL are loaded (deadlocks otherwise).
* The validation split must come from the base demos only (`--extra-train` is train-only); changing the dataset size
  would otherwise reshuffle validation into training.
* Text encoder `pretrained` needs `transformers` (now in the `train` extra); the frozen LM is not in checkpoints.
* Long jobs: run in the background and poll a log; a buffered pipe (`| grep`) hides progress until exit.

## 8. Working rules the user has set (keep following them)
* Report outcomes faithfully: say what failed and what was not run; no success claims without closed-loop evidence.
* Confirm before anything hard to reverse (force-push, deleting data/history). Never commit `data/` or `runs*/`.
* Keep new options backward compatible (defaults reproduce the previous behaviour) and add a test that fails when the
  guard is removed (mutate-and-check is the habit here).
* The user prefers short status updates with numbers, and decides direction; recommend, then act on their answer.

## 9. Unverified / assumptions to challenge
* Luxonis OAK-D Pro STL licence (repo is MIT; the STL is linked from a CDN) - confirm before redistributing.
* Real RGB sensor field of view (simulated as 70 deg vertical), the camera's mounting holes, the plate bolt pattern vs the
  xArm flange and VGC10 interface, the real VGC10 vacuum update rate/latency.
* Whether depth, or a seal/vacuum-level channel, would help (not tried; literature says depth can hurt on real sensors).
* The wedge under the camera is chunky (61 mm tall); a slimmer or hollowed bracket may be wanted.
