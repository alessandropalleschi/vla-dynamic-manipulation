# VLA for Dynamic Robotic Manipulation

**Goal:** a repeatable Docker-hosted Gazebo Harmonic + ROS 2 Jazzy simulation for stacking a carried box on a box moving along a conveyor. This is the environment and demonstration plumbing for a later belief-state VLA.

> **Status: integration starter / not tested against your Ubuntu + RTX 4070 machine.** The files were statically checked; the authoring environment had no Docker/Gazebo/NVIDIA runtime to execute a full simulation. The example expert is an *open-loop*, *simulation-only* scripted proof of concept, not verified grasping or a learned policy. Expect to tune control speed, contact parameters, camera framing and initial timing after first launch.

## What's included

- Gazebo Harmonic physics world with a moving **contact-driven** conveyor (TrackController), a lower target box, pickup tray, upper box and a fixed external RGB camera.
- A simple **3-axis Cartesian gantry** rather than a Panda **for the first slice**. This reduces robotics-control dependencies so you can test visual occlusions, moving-target contact and ROS middleware first. A Panda/MoveIt replacement is explicitly a follow-on step.
- Gazebo `DetachableJoint` for a **simulation-only pre-engaged vacuum grasp**: Harmonic gz-sim8 attaches at startup, so the tool starts aligned with the box. The demo lifts, tracks, lowers, releases and retracts. It does **not** learn the pick or prove suction physics.
- `ros_gz_bridge` for joint states, conveyor speed, gripper state, simulation clock and command topics; `ros_gz_image` for camera frames.
- A timed moving-target expert script and ROS bag (MCAP) recorder.
- Static project validator and ROS run-time topic smoke test.

## Prerequisites

Your confirmed hardware/software: Ubuntu 24.04.4, 61 GiB RAM, GeForce RTX 4070 12 GB, Docker 28.3.3, NVIDIA Container Toolkit 1.19.1. Driver 535.309.01 may require an upgrade **only if GPU headless rendering fails**, depending on the driver/runtime pairing. Do not assume Docker GPU support solely because `nvidia-ctk` exists.

Check Docker GPU routing on the **host**:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
docker compose version
```

If Docker reports that the NVIDIA runtime is not available:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Re-run the GPU routing check.
```

## Troubleshooting: Docker Compose exec import fix (v1.2)

**Why v1.1 still failed:** `docker compose exec` starts a new process using the
image/container environment, not shell exports created inside the entrypoint.
Sourcing ROS in `entrypoint.sh` prepares the main service process but does *not*
prepare an independently started `docker compose exec tools /usr/bin/python3`.
This is a common Docker behavior, not a problem with your host Conda environment.

**Instant workaround (no rebuild):**

```bash
docker compose exec tools bash -c \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 -c "import rclpy; print(rclpy.__file__)"'
docker compose exec tools bash -c \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/smoke_test.py'
```

If the sourced import succeeds but the direct import fails, the **old image is
missing its baked ROS paths**. Version 1.2 sets `PATH`, `PYTHONPATH`,
`LD_LIBRARY_PATH` and ROS prefix paths in `docker/Dockerfile` and tests the
unsourced import **during the image build**.

To apply **v1.2 patch** over your existing project, extract the patch ZIP at
repository root, then force a fresh image build and container recreation:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d --force-recreate
docker compose exec tools /usr/bin/python3 -c 'import rclpy; print(rclpy.__file__)'
docker compose exec tools bash /workspace/scripts/diagnose.sh
```

If `source` still does not fix the import, check package installation inside
the tools container:

```bash
docker compose exec tools dpkg-query -W ros-jazzy-rclpy
docker compose exec tools ls /opt/ros/jazzy/lib/python3.12/site-packages/rclpy
```

Do not `pip install rclpy` into your host `(base)` environment. ROS Jazzy's
Debian packages and system Python are intended to be used inside the container.

## Troubleshooting: launch fix (v1.1)

This revision fixes the previous `AMENT_TRACE_SETUP_FILES: unbound variable` error
by sourcing ROS **before** enabling Bash `set -u`. It also installs `ros-jazzy-rclpy`
explicitly and checks the import at Docker build time. The tools service now runs
`sleep infinity` directly instead of launching a login shell. Use `/usr/bin/python3`
when running ROS Python scripts so a Conda or other Python installation cannot
mask Ubuntu's ROS packages.

To diagnose the image:

```bash
docker compose exec tools bash /workspace/scripts/diagnose.sh
```

After extracting this updated archive over an existing checkout, **rebuild** because
`docker/Dockerfile` changed:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
docker compose logs --tail=80 sim
```

Then verify:

```bash
docker compose exec tools /usr/bin/python3 /workspace/scripts/smoke_test.py
```

## Quick start

Extract the project ZIP. In the directory containing `compose.yaml`:

```bash
# 1: Build ROS 2 Jazzy + Gazebo Harmonic container (network required for build)
docker compose build

# 2: Run the headless physics + RGB camera simulation and a tools shell service
docker compose up -d

# 3: Read startup diagnostics
docker compose logs -f sim
```

In a **second terminal**:

```bash
# 4: Confirm ROS 2 topics, camera frames and joint feedback
#    (this command should print PASS after receiving all three)
docker compose exec tools /usr/bin/python3 /workspace/scripts/smoke_test.py

# 5: Save a real image from the simulated external camera to data/camera_snapshot.png
docker compose exec tools /usr/bin/python3 /workspace/scripts/save_camera_frame.py

# 6: In a fresh simulation, run the timed baseline stacking sequence
docker compose exec tools /usr/bin/python3 /workspace/scripts/run_expert.py --speed 0.04
```

The robot begins with the simulated suction gripper already engaged above the pickup table. The script lifts the green upper box, starts the conveyor, follows the orange lower box, lowers the upper box, detaches it and retracts. **The expert uses commanded conveyor speed and initial pose; it does not process the camera**, so success is not guaranteed until tested and tuned.

To reset scene and box positions, restart the simulator:

```bash
docker compose restart sim
```

To record a demonstration (in a **third terminal**) start recording **before** running the expert:

```bash
docker compose exec tools bash /workspace/scripts/record.sh
```

Stop recording with `Ctrl+C`. A timestamped MCAP bag is saved beneath `data/` and survives container restarts because it is bind-mounted.

Stop everything:

```bash
docker compose down
```

## ROS 2 interface contract

| ROS topic | Message | Direction | Meaning |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | Gazebo → ROS | Simulation time |
| `/camera/image_raw` | `sensor_msgs/Image` | Gazebo → ROS | External camera (~20 FPS, 640×480) |
| `/joint_states` | `sensor_msgs/JointState` | Gazebo → ROS | Gantry X/Y/Z joint feedback |
| `/conveyor/speed` | `std_msgs/Float64` | ROS → Gazebo | Tangential belt velocity, m/s |
| `/gantry/x_cmd` | `std_msgs/Float64` | ROS → Gazebo | Prismatic X joint position, meters |
| `/gantry/y_cmd` | `std_msgs/Float64` | ROS → Gazebo | Prismatic Y joint position, meters |
| `/gantry/z_cmd` | `std_msgs/Float64` | ROS → Gazebo | Prismatic Z joint position, meters |
| `/gripper/attach` | `std_msgs/Empty` | ROS → Gazebo | (Re)attach named upper box |
| `/gripper/detach` | `std_msgs/Empty` | ROS → Gazebo | Release upper box |
| `/gripper/attached_state` | `std_msgs/String` | Gazebo → ROS | Attached/detached event |
| `/expert/phase` | `std_msgs/String` | Demo script → ROS | Scripted phase annotation |

The *future* learned-policy API will subscribe to `/camera/image_raw`, `/joint_states`, prior commanded action and task text, maintain a persistent latent belief, then publish bounded robot actions. The learner must **never** use Gazebo privileged object poses during evaluation.

## Directory layout

```text
conveyor-belief-vla/
├── compose.yaml
├── docker/
│   └── Dockerfile
├── config/
│   └── bridge.yaml
├── simulation/worlds/
│   └── conveyor_station.sdf
├── scripts/
│   ├── entrypoint.sh
│   ├── start_sim.sh
│   ├── smoke_test.py
│   ├── save_camera_frame.py
│   ├── run_expert.py
│   ├── record.sh
│   └── validate_project.py
├── docs/
│   └── next_steps.md
└── data/                       # MCAP demonstration recordings
```

## Troubleshooting

**`smoke_test.py` reports no camera frames:** inspect `docker compose logs sim`, run `docker compose exec sim gz topic -l` and look for `/station/camera/image`. Check NVIDIA graphics runtime capabilities, EGL support, and OGRE2 errors. The ROS image bridge requires the Gazebo camera topic to exist. If this fails because of a driver issue, fix the NVIDIA/EGL integration; a software-rendering fallback is not guaranteed for OGRE2.

**`smoke_test.py` reports no joint states:** check `docker compose exec sim gz topic -l | grep gantry`, then confirm `/gantry/joint_state` is published. Compare the bridge YAML mapping.

**Boxes don't move on conveyor:** ensure the target rests on the blue surface, the conveyor speed is published after the bridge is running, and DART contact surface motion works with your build. Publish a manual speed command:

```bash
docker compose exec tools ros2 topic pub -1 /conveyor/speed std_msgs/msg/Float64 '{data: 0.04}'
```

**Joint commands do not move gantry:** check `gz topic -l` for `/gantry/*_cmd`, then compare Gazebo and ROS topics. If contact with the moving target fails, tune the scripted durations and PID/velocity settings in the SDF, rather than assuming a given trajectory is valid.

**Camera orientation doesn't show the packing station:** edit `external_camera` pose and RGB sensor pose in `simulation/worlds/conveyor_station.sdf`, then `docker compose restart sim`. First verify image existence; visual framing is not yet calibrated to the real external camera.

**DDS cross-container discovery fails:** both services use host networking, `ROS_DOMAIN_ID=42` and Fast DDS. Check host firewall, existing processes using domain 42, and `docker compose exec tools ros2 topic list`.

**Static project validation (no Docker needed):**

```bash
python3 scripts/validate_project.py
python3 -m py_compile scripts/*.py
```

## Important limitations / follow-up tasks

1. **No trained VLA or RSSM yet**; this is the environment and ROS demonstration vertical slice.
2. **No Panda arm yet**; swap gantry for Panda description + `gz_ros2_control` and a Cartesian/MoveIt control stack only after validating this simpler scene.
3. **Simulated suction begins engaged** because gz-sim8's `DetachableJoint` implementation starts attached; no force-based grasp acquisition is modeled.
4. **No autonomous trials/reset manager or success oracle yet.** Restarting the simulator resets the boxes. Add a contact/pose-based evaluation node and deterministic randomization next.
5. **No guarantees of stack stability** without local physics testing. DART, slip, controller gains, and release latency all matter.

See `docs/next_steps.md` for the proposed RSSM and policy milestones.

## Primary upstream references

- Gazebo Harmonic / ROS 2: https://gazebosim.org/docs/harmonic/ros2_integration/
- Native TrackController (gz-sim8): https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1TrackController.html
- Native conveyor SDF example: https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/conveyor.sdf
- DetachableJoint implementation: https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/detachable_joint/DetachableJoint.cc
- Headless rendering: https://gazebosim.org/api/sim/7/headless_rendering.html
- ROS Jazzy / Gazebo Harmonic binary compatibility: https://github.com/gazebosim/ros_gz

## Placement diagnostics (v1.3)

After completing an expert trial, inspect real simulated box poses using
`/simulation/lower_box/pose` and `/simulation/upper_box/pose`. These use
`gz::sim::systems::PosePublisher` bridged as `PoseStamped`. No Docker image
rebuild is needed when applying the v1.3 diagnostic patch over v1.2: the
world and ROS bridge configuration are bind-mounted.

```bash
docker compose restart sim
docker compose exec tools bash -c 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/run_expert.py --speed 0'
docker compose exec tools bash -c 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/inspect_stack.py'
```

Ground truth should **never** be included in the deployed policy's input.
The checker evaluates alignment and expected height only, not actual
contact support, tilt or persistent stability. A complete success metric
still needs those additional checks.
