# VLA for Dynamic Robotic Manipulation

A simulation framework for studying **Vision-Language-Action (VLA) learning** in dynamic robotic manipulation, with a focus on moving-object interaction and reasoning under visual occlusion.

## Overview

The environment features a UF850 robotic arm equipped with an OnRobot VGC10 vacuum gripper. The robot picks up objects and stacks them onto moving targets on a conveyor belt.

The project provides a scripted manipulation policy and an RGB demonstration-recording pipeline as a foundation for learning-based control.

## Tech Stack

- **ROS 2 Jazzy** — Robot control and communication
- **Gazebo Harmonic** — Physics simulation
- **MoveIt 2** — Motion planning
- **Docker** — Reproducible environment
- **Python** — Control, data collection, and learning

## Quick Start

Run a moving-target stacking experiment:

```bash
bash scripts/run_uf850_moving_stack_trial.sh
```

Record an RGB demonstration:

```bash
bash scripts/record_and_run_uf850_v9.sh
```

## Research Directions

- Dynamic manipulation and moving-target interception
- Vision-based object tracking
- Learning from expert demonstrations
- Belief-state estimation under visual occlusion
- Vision-Language-Action policies

## Status

The simulation supports moving-conveyor stacking and synchronized RGB demonstration recording. Learning-based control and occlusion-aware belief estimation are under development.
