# UF850 + OnRobot VGC10 — v7 guarded stationary stacking test

This patch is for your **existing** `~/conveyor-belief-vla` ROS 2 Jazzy / Gazebo
Harmonic / MoveIt 2 installation. It does not overwrite the UF850 description,
Vacuum plugin, Gazebo world, Compose overlays, or existing v6 planner.

This is **experimental simulation software**, not a validated safety system.
It only works when the orange box and conveyor are stationary at the original SDF
poses, with no robot or tool geometry changes. **Never connect to the physical
UF850** or an ROS graph that can reach real hardware. An explicit acknowledgement
is not proof of isolation from hardware.

## Files

- `scripts/uf850_stationary_stack_v7.py`: 6D incremental TCP IK, per-stage MoveIt
  plan-only collision checking, explicit simulator controller execution, seal and
  attachment checks, carrying-box planning scene, planned trajectory and trace
  exports. Dry-run is the default (`--mode inspect`).
- `scripts/check_gz_box_pose_v7.py`: read-only *live Gazebo* green/orange box pose
  check using their SDF PosePublisher topics; saves a 60-second snapshot token.
- `scripts/verify_gz_stack_v7.py`: read-only two-sample physical stacking outcome check.
- `scripts/test_uf850_v7_offline.py`: synthetic geometry, IK, collision-matrix
  preservation, Gazebo text parser, and previously uploaded JSON-format tests.

The files use `scripts/uf850_moveit_common.py` from the already-installed v6
MoveIt patch and `/usr/bin/python3` ROS packages in the `moveit` image.

## 0. Install, inspect and test — no motion

Extract `uf850_v7_stationary_stack.zip` into the project root, preserving the
`scripts/` and `docs/` directories:

```bash
cd ~/conveyor-belief-vla
unzip -o ~/Downloads/uf850_v7_stationary_stack.zip

C='docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml'
$C up -d moveit
$C exec moveit bash -lc 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/test_uf850_v7_offline.py'
$C exec moveit bash -lc 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/uf850_stationary_stack_v7.py --mode inspect'
```

Stop the belt using its SDF-defined Gazebo transport topic:

```bash
$C exec sim bash -lc 'source /opt/ros/jazzy/setup.bash && gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"'
```

**Check the orange box is actually stopped**, not just that a zero-speed command
was sent. If it has drifted, reset the world/boxes to their SDF initial poses
before proceeding.

## 1. Confirm live box poses against SDF — read-only

Run *inside `sim`*, which has `gz` access to the actual running world:

```bash
$C exec sim bash -lc 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py'
```

Expected nominal centers (in Gazebo world coordinates):

- Green upper_box: `(-0.55, -0.55, 0.82)` meters.
- Orange lower_box: `(-0.52, 0.00, 0.83)` meters.

The check requires each box to be within 5mm XY, 6mm Z and ~0.05rad rotation of
its SDF pose; it saves `/workspace/data/uf850_gz_pose_snapshot.json` only on
success. If `gz topic` is not visible in the `sim` container, this check will
block rather than guessing. Retry after confirming `/simulation/upper_box/pose`
and `/simulation/lower_box/pose` are publishing.

## 2. Execute the already-tested hover plan — simulated movement begins

The prior user test reported **PASS** for live clock, single joint-state
publisher, vacuum OFF, controller action available, and exact start match.

**Before proceeding:** visually review the scene geometry, check conveyor
stopped and robot/tool clearances, and make sure your ROS domain 42 is isolated
from any hardware. This does not replace a MoveIt/RViz visual collision review.

```bash
$C exec moveit bash -lc '
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 /workspace/scripts/uf850_hover_guard.py \
    --urdf /workspace/simulation/generated/uf850_vgc10.urdf \
    --plan-file /workspace/data/uf850_hover_plan.json \
    --fresh-plan --execute \
    --ack-simulation-only --ack-conveyor-stopped --ack-scene-verified
'
```

The guard **replans immediately before the hover action**, checks the planned
start against live `/joint_states`, and monitors the ros2_control action result.
No vacuum is activated. It should finish with:

```
PASS: simulator arm reached hover goal, maximum joint error ... rad
VACUUM STAYS OFF. Verify gripper pose in both cameras before descent.
```

**Inspect the side camera at hover.** Four cups should be above the green box;
there should be no intersection with the pickup table. If the robot moved
unexpectedly, stop and investigate instead of continuing.

## 3. Plan the first approach — NO robot motion

At the **executed** hover pose (not home):

```bash
$C exec moveit bash -lc '
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 /workspace/scripts/uf850_stationary_stack_v7.py \
    --mode plan-contact
'
```

This uses corrected TCP+180° orientation from the generated URDF. It solves
near the current joint state for a pre-contact point **35 mm above the green
box top**, then calls MoveIt in **plan-only mode**. It writes
`data/uf850_stack_v7/01_precontact.json` on success. No trajectory is executed.

The script checks the hover pose is approximately above the pickup box,
rejects unintended low TCP positions along the approach, and preserves the
five static collision obstacles.

## 4. Run the full stationary stack experiment — simulation only

Right **before** executing, rerun Step 1 to refresh the Gazebo pose token,
then proceed ONLY if you have reviewed the hover and pre-contact plan and the
robot still occupies the verified hover pose.

```bash
$C exec sim bash -lc 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py'

$C exec moveit bash -lc '
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 /workspace/scripts/uf850_stationary_stack_v7.py \
    --mode execute-stack \
    --ack-simulation-only \
    --ack-conveyor-stopped \
    --ack-scene-verified \
    --ack-robot-hover-verified
'
```

The script plans EACH segment immediately before execution:

1. **Precontact** — hover to 35 mm above green top.
2. **Seal** — descend toward a 4 mm gap above green top; require positive
   `down_dot`, correct height/XY gap and **all four contact flags**.
3. **Vacuum on** — enable channels A and B and wait for Gazebo `/vgc10/attached`.
   If no attachment occurs, it **does not lift**.
4. **Lift** — allow only the attached green-box vs pickup-table contact while
   planning the initial 150 mm vertical lift. Robot/table collision checking is
   *never disabled*. Restore the collision-pair exception after this segment.
5. **Transfer** — plan with the green box **attached to `vgc10_tcp`** to a safe
   hover above the stationary orange box.
6. **Place** — lower the held green box to roughly 10 mm above the orange top.
7. **Release** — vacuum off, confirm Gazebo reports detached.
8. **Retreat** — plan 150 mm upward and restore the green box as a world
   collision object, then save all action trace data.

The commands are designed to stop on failed IK, failed collision-aware MoveIt
planning, stale state, controller rejection, loss of attachment while carrying,
missing cup seals, or excessive endpoint error. **They have NOT been exercised
against your running Gazebo; offline tests cannot guarantee the behavior.**

## Outputs and success criteria

Successful completed stages produce joint trajectories:

```
data/uf850_stack_v7/01_precontact.json
.../02_seal.json
.../03_lift.json
.../04_transfer.json
.../05_place_near.json
.../06_retreat.json
.../run_trace.jsonl
```

`run_trace.jsonl` logs ROS timestamp, joint feedback, vacuum state, diagnostics,
contact flags and stage. It does not contain the RGB frames needed for VLA.

**The script prints "STATIONARY STACK MOTION SEQUENCE COMPLETED" only after all
controller actions and detach checks succeed.** That message is NOT proof of a
stable stack: visually inspect the green box resting on the orange box and
check Gazebo's actual `/simulation/upper_box/pose` and
`/simulation/lower_box/pose`. The orange box must remain stationary for the
entire trial.

Run the **independent Gazebo physics outcome check** from the `sim` container:

```bash
$C exec sim bash -lc 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py'
```

It samples green and orange world poses twice, separated by 2 seconds, and
checks the green center is above the orange center by roughly 0.07m and aligned
in XY while both boxes are stationary. Its report is saved to
`data/uf850_stack_v7/stack_verification.json`. A successful result confirms the
**stationary simulator geometry** only, not a physical vacuum grasp, physical
stack stability, or moving-conveyor success.

## Failure recovery

- **If IK or collision planning fails:** the code does not execute that stage;
  inspect the named `BLOCKED:` reason. Never bypass collision checking.
- **If vacuum attaches and a later stage aborts:** the script intentionally
  leaves vacuum ON, rather than dropping a green box above the conveyor.
  Safely support the box before manually releasing vacuum. Cancelled
  trajectories can stop abruptly; observe the arm and reset the simulation.
- **If a fresh pose check fails:** the orange or green box has moved. Reset
  Gazebo to the initial poses; the v7 stationary planner must not be used with
  the conveyor moving.
- **If the workstation is connected to real hardware:** DO NOT run this code.
  ROS_DOMAIN_ID=42 is not a safety barrier; network isolation must be real.

## What is still outside this patch

Moving-conveyor tracking, variable orange-box pose updates, dynamically
replanning for a moving intercept, synchronous overhead RGB recording, and a
trained recurrent belief-state VLA policy remain subsequent milestones. A
successful static stack and a logged demonstration are the baseline for those
extensions; do not claim they are accomplished merely because this package
passes synthetic offline tests.
