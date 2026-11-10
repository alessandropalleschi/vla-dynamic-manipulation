#!/usr/bin/env python3
"""Host-side reproducible batch runner for the *working gantry* expert.

For each episode: reset sim -> collect images/actions/truth -> MP4 -> metrics.
This is intentionally NOT a UF850 trainer or a full physics contact evaluator.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, log_path=None, timeout=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    if log_path:
        with Path(log_path).open('w', encoding='utf-8') as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    else:
        p = subprocess.run(cmd, timeout=timeout)
    return p.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--speeds', default='0.0,0.02,0.04,0.06')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--postroll-s', type=float, default=4.0)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    speeds = [float(x.strip()) for x in a.speeds.split(',')]
    if not speeds or not all(0 <= s <= .10 for s in speeds):
        p.error('--speeds must be comma-separated values from 0.0 to 0.1 m/s')
    if a.episodes < 1 or a.episodes > 1000:
        p.error('--episodes must be 1..1000')
    if not 1 <= a.postroll_s <= 30:
        p.error('--postroll-s must be 1..30')
    if not shutil.which('docker'):
        p.error('Docker CLI not found')
    rng = random.Random(a.seed)
    batch = a.out or Path('data') / f'batch_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    if not batch.is_absolute():
        batch = ROOT / batch
    batch.mkdir(parents=True, exist_ok=True)
    (batch/'batch_config.json').write_text(json.dumps({
        'episodes':a.episodes,'speeds':speeds,'seed':a.seed,'postroll_s':a.postroll_s,
        'robot':'gantry', 'note':'UF850 arm not used by this expert'
    }, indent=2)+'\n')
    comp = ['docker','compose']
    if run([*comp, 'up', '-d', 'sim', 'tools']) != 0:
        print('Could not start sim/tools', file=sys.stderr)
        return 2
    overview = []
    for i in range(a.episodes):
        speed = rng.choice(speeds)
        epi = batch / f'episode_{i:04d}'
        epi.mkdir(exist_ok=True)
        print(f'\n=== EPISODE {i+1}/{a.episodes}  speed={speed:.3f} m/s ===', flush=True)
        rc = run([*comp, 'restart', 'sim'], log_path=epi/'restart.log', timeout=45)
        if rc == 0:
            # Allow bridge/gz sensor services to launch. Collector also waits for images+poses.
            time.sleep(4)
            command = [*comp, 'exec', '-T', 'tools', 'bash','-c',
                'source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 /workspace/scripts/collect_episode.py '
                ' --speed "$1" --out-dir "$2" --postroll-sim-s "$3"',
                '_', str(speed), '/workspace/'+str(epi.relative_to(ROOT)), str(a.postroll_s)]
            try:
                rc = run(command, log_path=epi/'collector.log', timeout=160)
            except subprocess.TimeoutExpired:
                rc = 124
        else:
            rc = 125
        print(f'Collector exit: {rc}', flush=True)
        if (epi/'frames').exists():
            run([sys.executable, str(ROOT/'scripts/encode_frames.py'), str(epi)],
                log_path=epi/'encode.log', timeout=120)
        run([sys.executable, str(ROOT/'scripts/evaluate_episode.py'), str(epi)],
            log_path=epi/'evaluate.log', timeout=15)
        result_path = epi/'evaluation.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            'verdict':'UNVERIFIED', 'reason':'evaluation script failed'
        }
        result.update(index=i, collector_exit=rc, requested_speed_m_s=speed)
        overview.append(result)
        print(f'  -> {result["verdict"]}: {result["reason"]}', flush=True)
        fields = sorted({key for row in overview for key in row})
        with (batch/'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(overview)
    successes = sum(r['verdict'] == 'POSE_STABLE' for r in overview)
    print(f'\nFinished batch: {successes}/{len(overview)} POSE_STABLE (contact unverified)')
    print(f'Summary: {batch/"summary.csv"}')
    # Keep researcher in control of what labels can be used for policy learning.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
