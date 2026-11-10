# VGC10 vacuum simulation v4 — UF850 / ROS 2 Jazzy / Gazebo Harmonic

This is a **simulation-only** four-cup VGC10 *geometric suction proxy*, not a vendor driver and not a physically calibrated model of negative pressure, deformable cups, surface seal, airflow, torque, or vacuum loss.

## Verified inputs from your workstation

- UF850 controller `uf850_arm_controller` is active and a +0.08 rad joint-1 smoke test passed.
- `/joint_states` now has a single authoritative `joint_state_broadcaster` publisher.
- UF850 URDF contains four proxy VGC10 cups in a 2×2 array. **Provisional dimensions**: diameter 30 mm and cup pitch 50 × 50 mm.

## Install

From the **project root** on the Ubuntu host, unpack this overlay and run:

```bash
bash scripts/build_uf850_vacuum.sh
```

This command:

1. Creates a backup of `docker/Dockerfile.uf850` before adding Gazebo C++ development dependencies (first run only).
2. Patches only the **generated UF850 world**, never `simulation/worlds/conveyor_station.sdf` (gantry reference).
3. Adds topics to your existing **UF850-specific** bridge file; it refuses to modify the gantry bridge file. This preserves the one-publisher `/joint_states` fix.
4. Builds the Docker image and the C++ Gazebo system plugin inside Docker.
5. Creates `compose.uf850.vacuum.yaml` so the sim container can locate the `.so` file.

Then start:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml up -d --force-recreate
docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml logs --tail=80 sim
```

Check the gripper without commanding robot motion:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml \
  exec tools bash -c 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/vgc10_vacuum_smoke.py'
```

The smoke test turns both vacuum channels on, observes the contact quality and attached state, turns both off, and confirms detachment. **It does not move the arm.** If the tool is away from the box, both channels on must not create an attachment.

## ROS 2 interface

| Topic | Type | Direction | Meaning |
|---|---|---|---|
| `/vgc10/channel_a/enable` | `std_msgs/msg/Bool` | ROS → Gazebo | Enable channel A (cups a1,a2) |
| `/vgc10/channel_b/enable` | `std_msgs/msg/Bool` | ROS → Gazebo | Enable channel B (cups b1,b2) |
| `/vgc10/attached` | `std_msgs/msg/Bool` | Gazebo → ROS | Whether fixed grasp constraint exists |
| `/vgc10/cup_contacts` | `std_msgs/msg/String` | Gazebo → ROS | e.g. `a1=1,a2=1,b1=0,b2=1`; **geometric contact proxy**, not contact-sensor data |
| `/vgc10/grasp_quality` | `std_msgs/msg/Float64` | Gazebo → ROS | Active qualified cup fraction, 0–1 |
| `/vgc10/status` | `std_msgs/msg/String` | Gazebo → ROS | WAITING, SEAL_WAIT, ATTACHED, etc. |
| `/vgc10/diagnostics` | `std_msgs/msg/String` | Gazebo → ROS | XY error, tool-to-box top gap, tool-down orientation dot |

## Attachment criterion

- Parent link: **UF850 `link6`**. Tool frame offset matches v3 default proxy: `link6` → suction plane `0 0 -0.097` m. **Validate tool mounting orientation** before relying on this.
- Four cups: centers at ±25 mm in local X/Y; effective cup radius 15 mm.
- Both channels must be enabled, tool must point down, and **at least three cup centers** must geometrically overlap the box's top surface (cup disk within top footprint), with an absolute gap ≤ 7 mm, for ≥ 0.15 s.
- Only then is a *physics* `components::DetachableJoint` fixed constraint created at runtime (not from startup).
- Disabling either channel releases the grasp (intentionally conservative proxy); box is then free to fall/move under Gazebo physics.
- The tool pose is derived from the actual simulated `link6` transform. Ground truth is **never intended as a policy observation**; it is internal to the vacuum simulator.

### Inspect alignment before planning

```bash
docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml \
  exec tools bash -c 'source /opt/ros/jazzy/setup.bash && ros2 topic echo /vgc10/diagnostics --once'
```

`height_gap_m` approximately zero and `xy_error_m` small indicates cup-to-box alignment. `down_dot` near +1 means the tool points down. Robot movement is deliberately **not automated by this patch**—an IK/planning controller is the next milestone. Start by verifying that the gripper state and alignment logic works.

## Caveats

- Gazebo Harmonic's standard DetachableJoint starts attached. This plugin creates / removes the component at runtime instead, so the box starts free.
- **No pressure physics or real contact sensors** in this version; contact is inferred from relative geometry. The `cup_contacts` field is not physical contact feedback.
- No validated MoveIt collision avoidance or UF850 stationary-pickup trajectory yet. Do not connect this ROS graph to real robot hardware.
- If you re-run `scripts/setup_uf850.sh` and regenerate the UF850 world file, re-run `python3 scripts/install_uf850_vacuum.py` before restarting.
- If the plugin loads but won't attach, inspect `/vgc10/diagnostics`: tool orientation and TCP offset are the most likely causes; do **not** weaken conditions until you confirm geometry.
- If you get CMake compile errors or Gazebo plugin-load errors, share those logs; C++ compilation/runtime has not been tested here against your actual Gazebo image.

## Positive plugin integration test (no arm movement)

Once the plugin builds, validate **both** attachment and detachment with a miniature Gazebo fixture that places a mock `link6` above a small box:

```bash
bash scripts/test_vgc10_fixture.sh
```

This starts a separate Gazebo partition; it does **not** touch your actual UF850 controller. It checks: (1) initially detached, (2) geometric cup seal plus both channels enabled leads to attach, (3) switching channel A off leads to detach. This is the strongest basic acceptance test before integrating motion planning.
