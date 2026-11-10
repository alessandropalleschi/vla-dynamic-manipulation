# `scripts/` — legacy Gazebo / ROS 2 pipeline

These scripts drive the original Gazebo Harmonic + MoveIt 2 stack (see `docker/`, `compose*.yaml`,
`simulation/`). They hardcode `/workspace/...` container paths, so they are kept in place.

New work (simulation, data, VLA policy) lives in the `vla_mujoco/` package; see `docs/MUJOCO.md`.
Version notes for the legacy stack are in `docs/legacy/`.
