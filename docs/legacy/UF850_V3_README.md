# UF850 + VGC10 simulation branch (v3 integration preview)

This is a **separate Gazebo Harmonic scene** based on the existing working gantry conveyor demo. It does NOT overwrite the baseline. Uses the official UFACTORY UF850 xArm ROS 2 `jazzy` description and installs a simplified OnRobot VGC10 tool with **four cups, 2×2**.

**Verified here:** source scripts pass offline Python and shell syntax checks; a synthetic six-joint URDF passes tool and gz_ros2_control composition tests. **Not verified here:** Docker startup, rendered meshes, arm dynamics, IK or physical vacuum forces. Your first acceptance test is six-joint feedback and a small trajectory. **Do not run the gantry expert or batch collector on this UF850 scene:** it assumes Cartesian gantry joints.

## Installation on Ubuntu 24.04 / ROS 2 Jazzy / Docker

From your existing `~/conveyor-belief-vla` directory, unzip the v3 patch (preserving the working baseline). Then:

```bash
cd ~/conveyor-belief-vla
bash scripts/setup_uf850.sh
```

The setup script:

1. Clones `https://github.com/xArm-Developer/xarm_ros2.git` **branch `jazzy`** under `external/xarm_ws/src/xarm_ros2/` (requires internet), records its commit.
2. Builds a separate Docker image with `gz_ros2_control`, `xacro`, and controller manager.
3. Builds **only** the `xarm_description` ROS package inside Docker.
4. Expands the official UF850 Xacro (six joints, no UFACTORY gripper) and appends VGC10 geometry.
5. Replaces Gazebo Classic / real robot control plugins with `gz_ros2_control/GazeboSimSystem` and a simulated six-joint trajectory controller.
6. Creates `simulation/worlds/conveyor_station_uf850.sdf` based on the current conveyor scene. The overhead camera and table are retained; the gantry is removed.

If the official xArm repo revision changes its Xacro structure or package dependencies, generation may fail. The error is left visible. Nothing connects to a real robot.

## Start / acceptance test

```bash
docker compose -f compose.yaml -f compose.uf850.yaml up -d
docker compose -f compose.yaml -f compose.uf850.yaml logs -f sim
```

After `UF850 ready for joint-level smoke tests` appears:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml exec tools bash -c \
  'source /opt/ros/jazzy/setup.bash && ros2 control list_controllers'
```

Expected: **`joint_state_broadcaster` and `uf850_arm_controller` active**.

Run a simulation-only test of a tiny **+0.08 rad motion of joint1**:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml exec tools bash -c \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/uf850_joint_smoke.py'
```

Expected: `PASS: six-joint UF850 feedback and commanded +0.08 rad joint1 rotation`.

Inspect the overhead image:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml exec tools bash -c \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/save_camera_frame.py'
xdg-open data/camera_snapshot.png
```

The initial spawn pose uses a pedestal center `x=-0.05, y=-0.47`, with the UF850 base `z=0.78 m`. **This is a starting scene layout; manually review reachability, clearance, and collisions before planning.**

## Important physical/controls limitations

- VGC10 dimensions in this project are an approximate **visual and collision geometry**, **not** manufacturer CAD. Cup diameter = **30 mm (provisional)** and pitch = **50×50 mm (provisional)**; change them after measurement.
- Four cups grouped in 2 channels (A: row 1, B: row 2) is an **assumption**, not validated plumbing. No vacuum pressure, cup sealing, contact-sensitive pick, or gripper-specific Gazebo plugin is implemented.
- The controller operates the **simulated six UF850 joints**, not Cartesian trajectories. This patch does **not** implement MoveIt IK, TCP calibration, collision-free pickup, or moving-target stacking.
- Generated mesh URLs point at paths under `/workspace/external/xarm_ws/` **inside the mounted Docker filesystem**; if Gazebo displays missing meshes, inspect file paths and `GZ_SIM_RESOURCE_PATH`.
- For the VLA dataset, the baseline gantry episodes remain a different embodiment. Do not mix the two action formats without an explicit embodiment conversion.

## How to return to the gantry baseline

```bash
docker compose -f compose.yaml -f compose.uf850.yaml down
docker compose up -d
```

The original `simulation/worlds/conveyor_station.sdf`, `scripts/run_expert.py`, and batch collection commands are untouched.

## Diagnostics

```bash
# Full simulation startup trace
 docker compose -f compose.yaml -f compose.uf850.yaml logs --tail=180 sim

# Controller interfaces
 docker compose -f compose.yaml -f compose.uf850.yaml exec tools bash -c \
  'source /opt/ros/jazzy/setup.bash && ros2 control list_hardware_interfaces'

# Spawned model in Gazebo
 docker compose -f compose.yaml -f compose.uf850.yaml exec sim bash -c \
  'source /opt/ros/jazzy/setup.bash && gz model -l'
```

Next robotics milestones: verified arm base pose/collision geometry → MoveIt 2 + UF850 IK → suction-cup collision/contact detection → two-channel A/B vacuum logic → moving-target expert for UF850 → automation and belief-policy training.
