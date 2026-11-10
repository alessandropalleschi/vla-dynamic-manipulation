# UF850 + VGC10 — v6 MoveIt 2 plan-only milestone

This is an **experimental, simulation-only**, collision-aware planning probe, not
an autonomous pickup controller. It does **not** send commands to
`uf850_arm_controller`, call the VGC10 vacuum topics, or execute motion.

## Preconditions

- Existing UF850 + VGC10 Gazebo simulation runs with one `/joint_states`
  publisher, `joint_state_broadcaster`, and the `uf850_arm_controller` is active.
- The **current corrected** `simulation/generated/uf850_vgc10.urdf` is present
  (after the 180-degree mount fix and the STL visual update).
- The `conveyor_station_uf850.sdf` file exists and matches the live scene.
- **Stop the conveyor** and reset to known initial box poses for this first
  plan-only test: world obstacles are a snapshot read from the SDF, *not* live
  Gazebo poses. A later milestone must refresh moving obstacles dynamically.

## 1. Install overlay

Extract the v6 ZIP **from the repository root**, preserving the directory
structure. This adds new files; it deliberately does not overwrite the existing
UF850, VGC10, gazebo or gantry scripts.

```bash
cd ~/conveyor-belief-vla
unzip -o ~/Downloads/uf850_moveit_v6_plan_only.zip
```

## 2. Check the candidate and world geometry without ROS

```bash
python3 scripts/plan_uf850_pickup.py
```

This prints the joint limits, valid 2π-equivalent representations of the IK
candidate, and the collision boxes to be sent to MoveIt. If a joint has no
allowed equivalent, it **rejects the target**. No clamping or robot movement.

## 3. Build the CPU-only MoveIt container

```bash
COMPOSE='docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml'
$COMPOSE build moveit
$COMPOSE up -d moveit
$COMPOSE logs -f --tail=80 moveit
```

The planner runs as a **separate container** on the existing ROS domain 42;
Gazebo and `uf850_arm_controller` remain in their already working containers.
The container installs ROS 2 Jazzy MoveIt 2 packages. GPU is not needed for
joint-space OMPL planning.

Look for `You can start planning now!` or equivalent `move_group` startup
messages. If the node exits, inspect its full logs; this milestone was not
runtime-tested against your exact UF850 installation.

## 4. Submit the PLAN-ONLY collision-aware goal

With Gazebo running and the conveyor **stopped**, run:

```bash
$COMPOSE exec moveit bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   /usr/bin/python3 /workspace/scripts/plan_uf850_pickup.py --plan'
```

If MoveIt returns SUCCESS, the planned trajectory (not an executed trajectory)
is saved to `data/uf850_hover_plan.json` on the host.

A planning failure is **useful information**. It could mean the start state is
in collision, the goal state is blocked, the joint limits are inconsistent, the
root-frame offset is wrong, the URDF/SRDF needs work, or the planner ran out of
search time. Do **not** bypass collision checking because of a failure.

## Coordinate-frame policy (read this)

The Gazebo UF850 is spawned at world XYZ `(-0.05, -0.47, 0.78)` while
`robot_state_publisher` currently uses the generated URDF rooted at its local
model frame. The initial SDF box obstacles are translated **by the negative of
that spawn offset** so they lie in the same robot-local frame as MoveIt. This
alignment is based on the current project scripts and should be checked
visually with TF/RViz before any execution.

The code includes the pickup table, conveyor surface, pedestal, initial green
box and orange box in the planning scene. It does not yet model the moving belt
frame, variable box poses, all fixtures, or gripper deformation. The simple
self-collision exclusion list disables *only immediately connected links*;
it is not a fully sampled MoveIt Setup Assistant collision matrix.

## What this milestone proves — and does not prove

- **Can prove on your machine:** the hover joint target is within official URDF
  limits and MoveIt finds a collision-aware plan from live joint states to it.
- **Cannot prove:** cartesian suction cup alignment under load, successful
  pickup, path execution, clearance from every unmodeled fixture, or stable
  stacking on a moving conveyor.
- **Absolutely no real-robot use:** this is an experimental planning stack.

## Debugging

```bash
$COMPOSE logs --tail=200 moveit
$COMPOSE exec moveit bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 action list'
$COMPOSE exec moveit bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 service list | grep planning_scene'
```

When this passes, the next steps are MoveIt path visualization, verified TCP
calibration, safe robot trajectory execution *in Gazebo*, visual-and-contact
checked suction pickup, then motion-synchronized stacking and demonstration
recording.
