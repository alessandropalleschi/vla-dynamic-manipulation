# v2.1 — VGC10 four-cup 2x2 update

This is a **small overlay patch** on top of v2. It updates the tool geometry only. Existing dataset recording, Gazebo Z-joint lower-limit fix, camera calibration, conveyor physics and controller scripts are left intact.

Confirmed: four physical cups, 2x2. The VGC10 has **two vacuum channels**, not four separately controlled channels. Default two cups assigned to each channel. Real A/B routing can differ with the adapter position.

## Install

```bash
cd ~/conveyor-belief-vla
unzip -o ~/Downloads/conveyor_v2_1_4cup.zip
python3 scripts/visualize_vgc10_on_gantry.py --world simulation/worlds/conveyor_station.sdf
python3 scripts/test_vgc10_4cup.py
# The 4-cup visual proxy is loaded when Gazebo restarts:
docker compose restart sim
```

The `visualize_vgc10_on_gantry.py` command is **idempotent** and writes `conveyor_station.sdf.pre_vgc10_4cup_visual.bak` the first time.

If your real cups are a different diameter/spacing, pass e.g. `--cup-mm 40 --pitch-x-mm 55 --pitch-y-mm 55`. Changing the visual proxy does not change grasp dynamics in the baseline simulation. `config/vgc10.yaml` describes default layout metadata; the generator uses matching CLI defaults rather than reading the YAML automatically.

### URDF preview

See `docs/VGC10_UF850.md` for generating a combined UF850 + four-cup tool URDF from the official robot description.

### Observation from current overhead camera image

The conveyor and pickup region are in frame, but the overhead gantry structure obscures part of the pickup station. That is a useful occlusion demonstration, but **a permanently hidden pickup surface is not desirable for learning**. Later, consider a slightly offset overhead view or replace the gantry with the actual UF850 scene. No camera transform is changed by this patch.
