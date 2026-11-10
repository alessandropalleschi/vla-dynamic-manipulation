VLA DYNAMIC MANIPULATION — CLEAN PIPELINE + PER-EPISODE PICKUP REPLANNING

Copy scripts/ into your existing vla-dynamic-manipulation repo.

INSTALL:
    cd ~/vla-dynamic-manipulation
    /usr/bin/python3 scripts/install_clean_pipeline.py

CHECK RANDOMIZATION WITHOUT MOVING THE ROBOT:
    /usr/bin/python3 scripts/collect_demos.py --episodes 5 --dry-run

ONE RECORDED EPISODE:
    bash scripts/record_demo.sh

BATCH WITH RANDOM GREEN/ORANGE XY, BELT SPEED AND EARLY MOTION:
    /usr/bin/python3 scripts/collect_demos.py --episodes 5 --seed 42

IMPORTANT:
 - v7/v9 scripts are preserved and serve as internal implementation/dependencies.
 - The new expert_moving_stack.py never reads a saved hover trajectory.
 - Every episode: new 6D IK, fresh MoveIt 00_hover trajectory, then contact/seal/lift.
 - Four-cup geometry check retained, 0.60 scaling requested.
 - Randomized SDF is restored after each episode.
 - The green box randomization is XY only, with no yaw variation initially.
 - Belt moves during short observation preroll, stops during pickup and transfer,
   then moves again for measured v9 target interception.
 - Live Gazebo test is still required; stop on any failure to replan or grasp.
