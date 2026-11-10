# v2 — Automated conveyor demonstrations (gantry) + UF850/VGC10 integration assets

**Two distinct maturity levels**:

1. **Data collection and pose-only evaluation** runs against your previously working **gantry** simulation (now with a VGC10-sized **visual-only** tool proxy; attachment physics unchanged). It collects timestamped RGB images, robot joint states, commanded actions, true box poses, phase markers, logs and annotated videos. It varies conveyor **speed only**, not initial positions/lighting yet.
2. **UF850 + VGC10** contains a robot-URDF tool augmentor and a separate scene scaffold. The physical 6-DOF arm and independently switchable suction channels are **not yet controlled by the demo expert**. Do not run the gantry collector against that scene.

## Install (overlay onto your existing repo; no Docker rebuild)

```bash
cd ~/conveyor-belief-vla
unzip -o ~/Downloads/conveyor_v2_vgc10.zip
bash scripts/install_v2_camera.sh
# Stop all older demo controllers before doing experiments.
docker compose restart sim
```

Check `simulation/worlds/conveyor_station.sdf.pre_overhead.bak` for the saved original camera viewpoint. The installer **preserves your fixed gantry Z joint lower limit** and fails if the old 0.0 value remains.

## First verification

```bash
docker compose exec tools bash -c 'source /opt/ros/jazzy/setup.bash && ros2 topic echo /camera/image_raw --once --field width'
docker compose exec tools bash -c 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/save_camera_frame.py'
xdg-open data/camera_snapshot.png
```

The installer also replaces the gantry's simplified suction-tip visuals with a 101×100×100 mm VGC10-sized enclosure and two 30 mm cup visuals. **This is only a camera/occlusion proxy, not a working VGC10 vacuum model or a UF850 arm.**

The overhead camera is **nadir** (optical axis down), with the image roughly aligned along the conveyor's X direction and a field of view covering pickup (Y = -0.55) and belt (Y = 0). Camera is a simulated proxy: calibrate intrinsics and extrinsics before using a real external camera.

## Collect multiple episodes

```bash
python3 scripts/batch_collect.py --episodes 3 --speeds 0.0,0.04 --seed 42 --postroll-s 4
```

Each episode restarts Gazebo (reset) and runs the ground-truth feedback gantry expert with the belt remaining active for 4 simulated seconds after the expert finishes. A short batch can take several minutes of wall time.

Outputs:

```
data/batch_<timestamp>/
  batch_config.json
  summary.csv
  episode_0000/
    rgb/frame_000000.jpg     # unannotated VLA-visible RGB
    telemetry.csv            # camera-aligned actions + robot states + privileged true poses
    frames/frame_000000.png  # text-overlay frame
    demonstration.mp4        # MP4 with phase and state overlays
    phases.csv
    expert.log
    capture_status.json
    evaluation.json
    collector.log
```

`telemetry.csv` mixes policy inputs and privileged labels for convenience. **Before model training, split columns**: use `rgb_file`, joint state, action history and instruction as policy inputs; keep `lower_*` and `upper_*` exclusively for expert/auxiliary supervision and evaluation. Never provide privileged poses to a deployed VLA.

## Evaluation criteria

`POSE_STABLE`: expert finished and release was observed; final >=1s window has <22mm horizontal offset, <16mm vertical-height error, <12mm relative drift; target translation is checked when speed >0.015m/s. These are **provisional thresholds**, not physical contact validation. `FAIL`: geometric/evolution constraints are violated or expert aborted. `UNVERIFIED`: required telemetry missing. A pose-stable episode is **not proof of suction seal, normal force, collision-free contact, or real-world success**.

The script preserves failed experiments and their logs; it does not silently relabel them successful. For low-speed experiments, if the object travels off the finite belt before the end of postroll, shorten postroll or adjust starting positions.

## UF850/VGC10 assets

See [VGC10_UF850.md](VGC10_UF850.md). The official UF850 description is obtained separately from UFACTORY's `xarm_ros2`, and **VGC10 is modeled as a tool geometry proxy**. The gantry data collector deliberately does not pretend to train the UF850.

## Troubleshooting

- `rclpy` import error: start commands via `docker compose exec tools bash -c 'source /opt/ros/jazzy/setup.bash && ...'`.
- No true pose topics: inspect `config/bridge.yaml` and `docker compose logs --tail=100 sim`.
- Stuck Z joint: check lower limit < 0 in Gazebo SDF; your working fix was `-0.03`.
- Video creation fails: PNG frames are still saved; run `python3 scripts/encode_frames.py data/batch_<timestamp>/episode_0000` on the host. FFmpeg is used through stdin to avoid the previous file-path issue.
- Camera doesn't cover both stations: revert from backup and adjust camera pose/FOV in `configure_overhead_camera.py`.
