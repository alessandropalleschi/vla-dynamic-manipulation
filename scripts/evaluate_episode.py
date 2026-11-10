#!/usr/bin/env python3
"""Strictly pose-based evaluation (NOT physical contact validation).

Consumes camera-aligned telemetry from collect_episode.py.
A failed/incomplete controller run is never reported as a success.
Missing fresh pose telemetry is unverified, never silently successful.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

BOX_UP = 0.06
BOX_LOW = 0.08
EXPECTED_DZ = (BOX_UP + BOX_LOW) / 2


def evaluate(run_dir: Path, min_window=1.0):
    status_path = run_dir / 'capture_status.json'
    telem_path = run_dir / 'telemetry.csv'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    result = {
        'episode': run_dir.name,
        'speed_m_s': status.get('requested_speed_m_s'),
        'expert_exit_code': status.get('expert_exit_code'),
        'frames': status.get('camera_frames'),
        'verdict': 'UNVERIFIED',
        'reason': '',
        'max_xy_error_m': None,
        'max_height_error_m': None,
        'max_relative_drift_m': None,
        'observed_target_speed_m_s': None,
        'post_release_duration_s': None,
        'released': False,
    }
    if not telem_path.exists():
        result['reason'] = 'telemetry.csv missing'
        return result
    with telem_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        result['reason'] = 'No telemetry rows'
        return result
    complete = int(status.get('expert_exit_code', -1) or 0) == 0
    def valid(row):
        keys = ('sim_time_s', 'lower_x', 'lower_y', 'lower_z', 'upper_x', 'upper_y', 'upper_z')
        try:
            return all(row[k] != '' and math.isfinite(float(row[k])) for k in keys)
        except (KeyError, ValueError):
            return False
    good = [r for r in rows if valid(r)]
    if not good:
        result['reason'] = 'No synchronized, fresh upper/lower poses'
        return result
    # Use *confirmed* detached status, or the expert's first release-phase marker
    # if attachment status is not latched. If neither is observable, unverified.
    detached = [float(r['sim_time_s']) for r in good if r.get('gripper') == 'detached']
    if not detached:
        result['reason'] = 'No observed gripper-detached state'
        if not complete:
            result['verdict'] = 'FAIL'
        return result
    release_t = min(detached)
    result['released'] = True
    after = [r for r in good if float(r['sim_time_s']) >= release_t + 0.25]
    if len(after) < 8:
        result['reason'] = 'Too few synchronized post-release samples'
        if not complete:
            result['verdict'] = 'FAIL'
        return result
    t_end = float(after[-1]['sim_time_s'])
    tail = [r for r in after if float(r['sim_time_s']) >= t_end - min_window]
    span = float(tail[-1]['sim_time_s']) - float(tail[0]['sim_time_s'])
    result['post_release_duration_s'] = round(t_end - release_t, 4)
    if span < min_window * 0.7 or len(tail) < 8:
        result['reason'] = 'Insufficient final stability window'
        if not complete:
            result['verdict'] = 'FAIL'
        return result
    if not complete:
        result['verdict'] = 'FAIL'
        result['reason'] = f'Expert did not complete (code {status.get("expert_exit_code")})'
        return result
    xy = [math.hypot(float(r['upper_x'])-float(r['lower_x']),
                     float(r['upper_y'])-float(r['lower_y'])) for r in tail]
    dz_err = [abs((float(r['upper_z'])-float(r['lower_z']))-EXPECTED_DZ) for r in tail]
    rel = [(float(r['upper_x'])-float(r['lower_x']),
            float(r['upper_y'])-float(r['lower_y'])) for r in tail]
    drift = [math.dist(pair, rel[0]) for pair in rel]
    result['max_xy_error_m'] = round(max(xy), 5)
    result['max_height_error_m'] = round(max(dz_err), 5)
    result['max_relative_drift_m'] = round(max(drift), 5)
    v = (float(tail[-1]['lower_x'])-float(tail[0]['lower_x']))/max(span, .001)
    result['observed_target_speed_m_s'] = round(v, 5)
    moving = (result['speed_m_s'] or 0.0) >= 0.015
    target_moving = (abs(v) > 0.3*float(result['speed_m_s'])) if moving else True
    geometry_ok = max(xy) <= 0.022 and max(dz_err) <= 0.016 and max(drift) <= 0.012
    if geometry_ok and target_moving:
        result['verdict'] = 'POSE_STABLE'
        result['reason'] = 'Stack geometry stable in final window; contact NOT verified'
    else:
        result['verdict'] = 'FAIL'
        problems = []
        if max(xy) > 0.022: problems.append('horizontal offset > 22 mm')
        if max(dz_err) > 0.016: problems.append('height error > 16 mm')
        if max(drift) > 0.012: problems.append('relative drift > 12 mm')
        if not target_moving: problems.append('target not visibly moving despite speed command')
        result['reason'] = '; '.join(problems)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--stability-window-s', type=float, default=1.0)
    a = p.parse_args()
    result = evaluate(a.episode, a.stability_window_s)
    out = a.episode / 'evaluation.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'{result["verdict"]}: {result["reason"]}  -> {out}')
    return 0 if result['verdict'] == 'POSE_STABLE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
