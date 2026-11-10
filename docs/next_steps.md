# Next deliverables: from scripted simulation to learned belief policy

## Deliverable 2 — controllable episodes and physically validated baseline

1. Run `smoke_test.py` on the RTX 4070 host and fix any rendering / topic issues.
2. Inspect camera image and tune its extrinsics to match the actual packing station.
3. Replace timed open-loop x extrapolation with a feedback-based target-pose estimator **in the expert only**. Use Gazebo truth poses during training if needed.
4. Move from pre-engaged suction to a robot-actuated gripper. Replace the gantry with Panda once a `gz_ros2_control` stack is stable.
5. Build an episode manager with `/episode/reset` and deterministic seeds; log target pose and contact states separately as privileged labels.
6. Define success: upper-box support/contact, relative horizontal placement within tolerance and stack stability for 0.5s while belt moves.

## Deliverable 3 — learned belief and action policy

Observation: timestamped external RGB (possibly a small frame window), joint pose/velocity, gripper state, last action, elapsed time, and optional language instruction.

```
image_t -> frozen small encoder -> embedding_t ----+
robot_t, action_(t-1), dt --------------------------+--> RSSM update
                                                   |
                                         h_t + z_t (belief)
                                                   |
language embedding -------------------------------+--> action head
                                                   |
                                       bounded EE/joint action chunk
                                                   |
                                            ROS 2 command adapter
```

Use an action-conditioned RSSM with deterministic recurrent state `h_t` and stochastic latent `z_t`. Train observation/dynamics losses, auxiliary ground-truth pose/velocity and contact labels, plus uncertainty calibration. Inference sees **no privileged state**.

Start with recurrent behavior cloning; graduate to a small diffusion/flow-matching head if benchmark latency/memory permits. Keep action frequency and timestamping explicit.

## Proposed controlled comparison

- Single-frame BC (no memory)
- Fixed-frame-window BC
- Recurrent BC (implicit memory)
- RSSM + action policy (predictive belief)

Test occlusion length (0 / 0.25 / 0.5 / 1s), speed (0.02–0.1m/s), slippage, surprise speed change, initial box pose, camera viewpoint and latency. Measure success, time, contact events, horizontal placement error, target-position error during occlusion, and calibrated uncertainty.

## Notes about real-world deployment

A trained policy should command a bounded position/velocity interface, not directly bypass low-level controls. Add collision limits, joint limits, stop conditions and independent watchdogs. Avoid treating a hallucinated high-confidence belief during occlusion as guaranteed geometric ground truth.
