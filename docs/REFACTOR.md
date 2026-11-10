# Clean pipeline

Run one trial: `bash scripts/run_stack.sh`.

Record one episode: `bash scripts/record_demo.sh`.

Collect randomized demonstrations: `python3 scripts/collect_demos.py`.

Core scripts: `expert_moving_stack.py`, `record_episode.py`, `setup_camera.py`.

Legacy v7/v9 implementations remain temporarily for compatibility.

Every episode replans `00_hover` using live robot joints and the randomized green-box pose. No saved trajectory is required.
