#!/usr/bin/env python3
"""Generate RGB demonstrations with fresh pickup and moving-target plans.

Randomizes green pickup XY, orange conveyor target XY, belt speed and the
observation preroll. Restores the source SDF byte-for-byte after each run.
All motion is in isolated Gazebo simulation, never on hardware.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / 'simulation/worlds/conveyor_station_uf850.sdf'
OUTPUT = ROOT / 'data/vla_demos'


def models_from_sdf(raw):
    root = ET.fromstring(raw)
    boxes = {}
    for name in ('upper_box', 'lower_box'):
        pose = root.find(f"./world/model[@name='{name}']/pose")
        if pose is None or not (pose.text or '').strip():
            raise RuntimeError(f'SDF model {name} has no pose')
        values = [float(v) for v in pose.text.split()]
        if len(values) != 6 or any(abs(v) > 1e-6 for v in values[3:]):
            raise RuntimeError(f'Expected axis-aligned six-component pose for {name}')
        boxes[name] = (pose, values)
    return root, boxes


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episodes', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--speeds', default='0.015,0.020,0.025,0.030,0.040')
    ap.add_argument('--green-x-range', type=float, nargs=2, default=(-.565, -.535))
    ap.add_argument('--green-y-range', type=float, nargs=2, default=(-.565, -.535))
    ap.add_argument('--orange-x-range', type=float, nargs=2, default=(-.55, -.52))
    ap.add_argument('--orange-y-range', type=float, nargs=2, default=(-.006, .006))
    ap.add_argument('--preroll-mm-range', type=float, nargs=2, default=(40., 60.))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not 1 <= args.episodes <= 200:
        ap.error('--episodes must be 1–200')
    try:
        speeds = [float(v.strip()) for v in args.speeds.split(',')]
    except ValueError:
        ap.error('Invalid --speeds list')
    if not speeds or any(not (.012 <= v <= .040) for v in speeds):
        ap.error('All speeds must be 0.012–0.040 m/s')
    ranges = [
        (args.green_x_range, -.57, -.53, 'green X'),
        (args.green_y_range, -.57, -.53, 'green Y'),
        (args.orange_x_range, -.57, -.50, 'orange X'),
        (args.orange_y_range, -.01, .01, 'orange Y'),
        (args.preroll_mm_range, 20., 70., 'preroll travel'),
    ]
    for pair, lo, hi, name in ranges:
        if not lo <= pair[0] <= pair[1] <= hi:
            ap.error(f'Invalid {name} range, allowed [{lo}, {hi}]')

    if not WORLD.is_file():
        raise RuntimeError(f'Missing station world: {WORLD}')
    if not (ROOT / 'scripts/record_demo.sh').is_file():
        raise RuntimeError('Install the clean expert/recorder first')

    original = WORLD.read_bytes()
    _, boxes0 = models_from_sdf(original)
    rng = random.Random(args.seed)
    tag = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    speed_choices = [speeds[i % len(speeds)] for i in range(args.episodes)]
    rng.shuffle(speed_choices)

    configurations = []
    for i, speed in enumerate(speed_choices):
        gx = rng.uniform(*args.green_x_range)
        gy = rng.uniform(*args.green_y_range)
        ox = rng.uniform(*args.orange_x_range)
        oy = rng.uniform(*args.orange_y_range)
        travel_m = rng.uniform(*args.preroll_mm_range) / 1000
        preroll_s = travel_m / speed
        if preroll_s > 8:
            raise RuntimeError('Preroll >8 sim seconds; shorten range or raise min speed')
        if ox + travel_m + .030 > -.385:
            raise RuntimeError('Predicted interception outside tested conveyor region')
        label = f'{tag}_{i:03d}'
        conf = {
            'episode': label, 'seed': args.seed, 'episode_index': i,
            'green_initial_xyz_m': [round(gx, 6), round(gy, 6), boxes0['upper_box'][1][2]],
            'orange_initial_xyz_m': [round(ox, 6), round(oy, 6), boxes0['lower_box'][1][2]],
            'belt_speed_m_s': speed,
            'preroll_sim_s': round(preroll_s, 6),
            'preroll_nominal_travel_m': round(travel_m, 6),
            'velocity_scale': .60, 'acceleration_scale': .60,
            'fresh_pickup_ik': True, 'fresh_pickup_moveit_plan': True,
            'saved_hover_plan_used': False,
        }
        configurations.append(conf)
        print('\n===', label, '===', flush=True)
        print(json.dumps(conf, indent=2), flush=True)
        if args.dry_run:
            continue

        # Change world model spawn positions only during this run.
        root, boxes = models_from_sdf(original)
        for name, (x, y) in {
            'upper_box': (gx, gy), 'lower_box': (ox, oy)
        }.items():
            node, original_pose = boxes[name]
            modified = original_pose[:]
            modified[0], modified[1] = x, y
            node.text = ' '.join(f'{v:.9f}' for v in modified)

        folder = OUTPUT / label
        OUTPUT.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ,
                   VLA_EPISODE_NAME=label,
                   VLA_BELT_SPEED_M_S=str(speed),
                   VLA_WARMUP_SIM_S=f'{preroll_s:.6f}',
                   VLA_IK_SEED=str(args.seed * 1000 + i))
        result = None
        try:
            WORLD.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))
            with (OUTPUT / (label + '.launcher.log')).open('w') as log:
                result = subprocess.run(['bash', 'scripts/record_demo.sh'],
                                        cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
        finally:
            WORLD.write_bytes(original)
            if folder.is_dir():
                (folder / 'randomization.json').write_text(json.dumps(conf, indent=2) + '\n')
        conf['exit_code'] = result.returncode if result else -1
        print(f'EPISODE EXIT: {conf["exit_code"]} | log: data/vla_demos/{label}.launcher.log', flush=True)

    if args.dry_run:
        print(f'DRY RUN: {len(configurations)} parameter sets; nothing modified')
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / (tag + '.summary.json')
    report.write_text(json.dumps(configurations, indent=2) + '\n')
    success = sum(c.get('exit_code') == 0 for c in configurations)
    print(f'FINISHED: {success}/{len(configurations)} successful episodes | {report}')
    print('Original SDF restored byte-for-byte; restart Gazebo for unrandomized station.')
    return 0 if success == len(configurations) else 2


if __name__ == '__main__':
    try:
        sys.exit(run())
    except (RuntimeError, OSError, subprocess.SubprocessError, ET.ParseError) as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        sys.exit(2)
