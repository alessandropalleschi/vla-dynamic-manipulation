# Automatic camera video capture (v1.4)

Run from the **Ubuntu host**, in the project root:

```bash
sudo apt update && sudo apt install -y ffmpeg   # only if ffmpeg isn't installed
bash scripts/run_demo_video.sh 0.0               # stationary baseline
bash scripts/run_demo_video.sh 0.04              # moving conveyor baseline
```

Each run:
1. Starts Docker services if necessary and restarts Gazebo to reset the scene.
2. Subscribes to `/camera/image_raw` and waits for at least three camera images.
3. Records annotated camera frames to `data/demo_<timestamp>.../frames/`.
4. Runs `run_expert.py` automatically, recording `/expert/phase` transitions.
5. Records a two-second postroll (wall clock) and encodes a 20 FPS MP4 with host FFmpeg.
6. Adds `stack_report.txt` if the optional `inspect_stack.py` from v1.3 is present.

Output directories include:

```
data/demo_.../
  demonstration.mp4   # share this video for visual diagnosis
  frames.csv          # per-frame phase, gripper and simulation timestamps
  phases.csv          # phase changes and nearest frame number
  expert.log          # stdout / errors from run_expert.py
  keyframes/          # snapshots around phase transitions
  frames/             # individual annotated camera frames
  stack_report.txt    # when v1.3 pose diagnostics available
```

`data/latest_demo` is a symlink to the most recent run. Open it with
`xdg-open data/latest_demo/demonstration.mp4`.

**Timing caveat:** The video plays at a nominal 20 frames/second, matching the
simulated camera's configured update rate. Dropped camera frames or a simulation
running faster/slower than real time mean MP4 playback time is approximate.
The overlays and `frames.csv` contain actual camera simulation timestamps.

**Important:** This records the **simulated external RGB camera**, not the
Gazebo desktop GUI. It does not prove successful stacking. Geometry and contact
need separate ground-truth evaluation. Do not use privileged pose topics as
input to your learned policy.

No Dockerfile changes or rebuild are required for this patch. The recorder uses
ROS 2, `rclpy`, and Pillow already installed in the `tools` container. ROS is
sourced explicitly so the previously encountered `rclpy` import issue is avoided.
