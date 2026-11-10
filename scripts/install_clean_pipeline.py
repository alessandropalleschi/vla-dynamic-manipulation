#!/usr/bin/env python3
"""Install stable-named dynamic stacking/recording entrypoints.

Preserves the successful v9 and v7 source files. The exported expert replans
hover for *every* episode from live joints and the randomized SDF pose.

Run from the repository root: /usr/bin/python3 scripts/install_clean_pipeline.py
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
SCRIPTS = ROOT / 'scripts'
if not (ROOT / 'compose.yaml').is_file():
    sys.exit('BLOCKED: run from repository root (compose.yaml missing)')

V9 = SCRIPTS / 'uf850_moving_stack_v9.py'
EXPERT = SCRIPTS / 'expert_moving_stack.py'
if not V9.is_file() and not EXPERT.is_file():
    sys.exit('BLOCKED: working v9 script not found')

# Always prefer the verified legacy version on first installation, and the
# stable-named version on subsequent runs (both remain available).
s = (EXPERT if EXPERT.is_file() else V9).read_text()
for token in ('INTERCEPT READY:', 'BELT VERIFIED:', 'MovingStation'):
    if token not in s:
        sys.exit(f'BLOCKED: v9 source lacks verified {token!r} marker')

# Variable belt speed for recorded demonstrations. Keep measured-speed gate.
if re.search(r'^BELT_M_S\s*=\s*0\.02\s*$', s, re.M):
    s = re.sub(
        r'^BELT_M_S\s*=\s*0\.02\s*$',
        "BELT_M_S = float(os.environ.get('VLA_BELT_SPEED_M_S', '0.020'))\n"
        "WARMUP_SIM_S = float(os.environ.get('VLA_WARMUP_SIM_S', '0.0'))",
        s, count=1, flags=re.M,
    )
elif 'VLA_BELT_SPEED_M_S' not in s:
    sys.exit('BLOCKED: unable to configure moving belt speed')
if 'WARMUP_SIM_S' not in s:
    s = s.replace('MAX_X_TRAVEL_M =',
                  "WARMUP_SIM_S = float(os.environ.get('VLA_WARMUP_SIM_S', '0.0'))\n"
                  'MAX_X_TRAVEL_M =', 1)
if '0.65 * BELT_M_S < speed < 1.45 * BELT_M_S' not in s:
    old = '.012 < speed < .030'
    if s.count(old) != 1:
        sys.exit('BLOCKED: measured conveyor-speed check not recognized')
    s = s.replace(old, '0.65 * BELT_M_S < speed < 1.45 * BELT_M_S')

# Replace the saved-plan reader, not the actual motion-planning stage.
FRESH_HOVER = '''def plan_fresh_hover(station, model, targets):
    """Fresh 6D IK + MoveIt plan for the green box in this episode.

    The old validated joint pose is an *IK seed only*, NOT a saved goal or
    a trajectory. Every attempt solves the currently randomized green pickup
    pose and obtains a new collision-aware plan from live joint feedback.
    """
    current = station.check_fresh()
    green = station.world['upper_box']
    target = np.asarray(targets['pickup_seal'], dtype=float).copy()
    target[2] += 0.116
    v7.check(target[2] > green['center_world'][2] + green['size'][2] / 2 + .10,
             'Pickup hover does not clear the green box')

    # A known robot configuration is only a numerical starting guess for IK.
    # No file in data/ is read to choose or execute this hover.
    nominal = np.array([-2.987797158, -0.445483165, -0.930707982,
                         0.178675985, -0.460686831, -0.740066600])
    nominal = np.clip(nominal, model.lo_vec + 0.01, model.hi_vec - 0.01)
    base_R = model.T(nominal)[:3, :3]
    orientations = [base_R] + [v7.rot_rpy([0.0, 0.0, a])
        for a in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    rng = np.random.default_rng(int(os.environ.get('VLA_IK_SEED', '91')))
    seeds = [nominal, current, (nominal + current) / 2]
    seeds.extend(np.clip(nominal + rng.normal(0, .14, 6),
                         model.lo_vec + .01, model.hi_vec - .01)
                 for _ in range(9))

    # Four 40.4mm lip disks at 44mm pitch, with physical box dimensions.
    pitch, radius = .044, .0202
    cx, cy = np.asarray(green['center_world'][:2], dtype=float)
    half_x, half_y = np.asarray(green['size'][:2], dtype=float) / 2
    found = []
    for R in orientations:
        for seed in seeds:
            try:
                q = model.solve_near(seed, target, R,
                                     tol_pos=.0015, tol_angle=.017)
                T = model.T(q)
                if T[2, 2] < .995 or np.linalg.norm(T[:3, 3] - target) > .0015:
                    continue
                if any(
                    abs((T[:3, 3] + T[:3, :3] @ [x*pitch/2, y*pitch/2, 0])[0] - cx)
                        + radius > half_x + 1e-6
                    or abs((T[:3, 3] + T[:3, :3] @ [x*pitch/2, y*pitch/2, 0])[1] - cy)
                        + radius > half_y + 1e-6
                    for x in (-1, 1) for y in (-1, 1)
                ):
                    continue
                if any(np.linalg.norm(q - old[1]) < .01 for old in found):
                    continue
                found.append((float(np.linalg.norm(q - current)), q))
            except (v7.Blocked, ValueError, np.linalg.LinAlgError):
                continue
        if len(found) >= 5:
            break
    found.sort(key=lambda item: item[0])
    v7.check(found, 'No IK hover with all four cups fitting randomized green box')

    failures = []
    for _, q in found[:8]:
        try:
            plan = station.plan('00_hover', q, holding=False)
            print('PASS: Fresh hover IK and MoveIt plan for green XYZ',
                  np.round(green['center_world'], 4).tolist(), flush=True)
            return plan
        except (v7.Blocked, RuntimeError) as exc:
            failures.append(str(exc))
    raise v7.Blocked('MoveIt rejected all randomized hover candidates: '
                     + '; '.join(failures[-3:]))


'''
if 'def plan_fresh_hover(' not in s:
    m = re.search(r'^def choose_hover\([\s\S]*?(?=^def run\()', s, re.M)
    if not m:
        sys.exit('BLOCKED: cannot locate saved hover plan reader')
    s = s[:m.start()] + FRESH_HOVER + s[m.end():]

# Replace the stale qhover assignment and subsequent use.
s, removed = re.subn(
    r'^\s*qhover\s*=\s*choose_hover\(model,\s*args\.hover_plan,\s*t\)\s*$',
    '', s, count=1, flags=re.M,
)
if not removed and 'plan_fresh_hover(station, model, t)' not in s:
    sys.exit('BLOCKED: saved hover plan call not located')

pattern = r"hover\s*=\s*station\.plan\('00_hover_moving_trial',\s*qhover\)"
if re.search(pattern, s):
    s = re.sub(pattern, "hover = plan_fresh_hover(station, model, t)", s, count=1)
elif 'hover = plan_fresh_hover(station, model, t)' not in s:
    sys.exit('BLOCKED: hover planning call not located')
s = s.replace("station.move('00_hover_moving_trial', hover)",
              "station.move('00_hover', hover)")
s = s.replace("'STAGE 00: new collision-aware plan to validated hover joints'",
              "'STAGE 00: solve and plan a NEW pickup hover'")
# Remove now-unused --hover-plan interface for the stable script only.
s = re.sub(r"^\s*ap\.add_argument\('--hover-plan'[^\n]*\)\s*$", '', s, flags=re.M)

# Observation preroll: show belt motion to the camera before pickup.
anchor = "        v7.check(np.linalg.norm(p-world['lower_box']['center_world']) < .005,\n                 'Orange box not at reset pose; restart Gazebo before trial')"
if anchor not in s and 'RANDOMIZED_OBSERVATION_PREROLL' not in s:
    sys.exit('BLOCKED: initial orange-box pose check not found')
if 'RANDOMIZED_OBSERVATION_PREROLL' not in s:
    warmup = '''
        # RANDOMIZED_OBSERVATION_PREROLL
        # Observe moving target before pickup; stop to keep the validated
        # green pickup and transfer scene collision-aware and deterministic.
        v7.check(0.012 <= BELT_M_S <= 0.040,
                 'Belt speed must be in [0.012, 0.040] m/s')
        v7.check(0.0 <= WARMUP_SIM_S <= 8.0,
                 'Preroll must be in [0, 8] simulation seconds')
        if WARMUP_SIM_S > 0:
            start_clock = station.state['clock']
            v7.check(start_clock is not None, 'Missing Gazebo simulation clock')
            station.belt(True)
            try:
                deadline = time.monotonic() + max(25., WARMUP_SIM_S * 5.)
                while station.state['clock'] < start_clock + WARMUP_SIM_S:
                    v7.check(time.monotonic() < deadline,
                             'Gazebo clock stopped during preroll')
                    station.spin(.08)
            finally:
                station.belt(False)
            station.spin(.4)
            shifted = station.live_orange()
            dx = float(shifted[0] - p[0])
            v7.check(.003 < dx < .090,
                     f'Conveyor preroll had unexpected travel: {dx:.4f}m')
            v7.check(abs(shifted[1] - p[1]) < .006,
                     'Unexpected lateral drift during preroll')
            p = shifted
            station.world['lower_box']['center_world'] = p.copy()
            station.world['lower_box']['center'] = p - np.asarray(v7.SPAWN_WORLD)
            t = v7.stacking_targets(station.world)
            v7.check(p[0] + .030 <= -.385,
                     'Orange box left the tested intercept region')
            print('PREROLL PASS: orange stopped at',
                  np.round(p, 4).tolist(), flush=True)
'''
    s = s.replace(anchor, anchor + warmup)

# Internally named outputs should not refer to old experiments.
s = s.replace("default='/workspace/data/uf850_stack_moving_v8'",
              "default='/workspace/data/expert_runs/current'")
s = s.replace("Path('/workspace/data/uf850_stack_moving_v8')",
              "Path('/workspace/data/expert_runs/current')")

# Ensure fresh hover planning actually happens only AFTER ROS is ready.
try:
    expert_ast = ast.parse(s)
except SyntaxError as exc:
    sys.exit(f'BLOCKED: generated expert has invalid Python syntax: {exc}')
if 'choose_hover(' in s or 'qhover' in s:
    sys.exit('BLOCKED: saved-hover dependency survived refactor')
if s.count('plan_fresh_hover(station, model, t)') != 1:
    sys.exit('BLOCKED: expected exactly one fresh-hover plan call')

# Recorder kept intact except for stable expert and report paths.
RECORD_OLD = SCRIPTS / 'record_uf850_vla_episode.py'
RECORD_NEW = SCRIPTS / 'record_episode.py'
if not RECORD_OLD.is_file() and not RECORD_NEW.is_file():
    sys.exit('BLOCKED: current RGB demonstration recorder not found')
rec = (RECORD_NEW if RECORD_NEW.is_file() else RECORD_OLD).read_text()
if 'uf850_moving_stack_v9.py' not in rec and 'expert_moving_stack.py' not in rec:
    sys.exit('BLOCKED: cannot find expert invocation in recorder')
rec = rec.replace('uf850_moving_stack_v9.py', 'expert_moving_stack.py')
rec = rec.replace('uf850_stack_moving_v8', 'expert_runs/current')
# Isolate plans and traces per episode; never copy stale files from a past run.
expert_cmd = "                '--ack-simulation-only',"
if "'--reports', str(args.out / 'plans')" not in rec:
    if rec.count(expert_cmd) != 1:
        sys.exit('BLOCKED: cannot locate expert invocation for isolated episode reports')
    rec = rec.replace(expert_cmd,
                      expert_cmd + "\n                '--reports', str(args.out / 'plans'),")
rec = rec.replace("Path('/workspace/data/expert_runs/current')", "(args.out / 'plans')")
rec = rec.replace("Path('/workspace/data/expert_runs/current/run_trace.jsonl')",
                  "(args.out / 'plans' / 'run_trace.jsonl')")
rec = rec.replace("args.out / 'plans'.glob(", "(args.out / 'plans').glob(")
if 'observation_preroll_sim_s' not in rec:
    rec = rec.replace("'joint_velocity_scale': 0.6, 'joint_acceleration_scale': 0.6,",
        "'joint_velocity_scale': 0.6, 'joint_acceleration_scale': 0.6,\n"
        "        'conveyor_speed_m_s': float(os.getenv('VLA_BELT_SPEED_M_S', '0.020')),\n"
        "        'observation_preroll_sim_s': float(os.getenv('VLA_WARMUP_SIM_S', '0.0')),")
ast.parse(rec)

cam_old = SCRIPTS / 'install_uf850_overhead_camera.py'
cam_new = SCRIPTS / 'setup_camera.py'
if not cam_old.is_file() and not cam_new.is_file():
    sys.exit('BLOCKED: overhead camera installer missing')

# Compose can optionally apply the user's local UID/GID override.
common = '''#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
C=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml)
[[ ! -f compose.user.yaml ]] || C+=(-f compose.user.yaml)
stop_belt() { "${C[@]}" exec -T sim bash -lc 'gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"' >/dev/null 2>&1 || true; }
trap stop_belt EXIT
'''
launch = common + '''"${C[@]}" up -d --force-recreate sim moveit
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0"; /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py'
"${C[@]}" exec -T \\
  -e "VLA_BELT_SPEED_M_S=${VLA_BELT_SPEED_M_S:-0.020}" \\
  -e "VLA_WARMUP_SIM_S=${VLA_WARMUP_SIM_S:-0}" \\
  -e "VLA_IK_SEED=${VLA_IK_SEED:-91}" \\
  moveit bash -lc 'source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/expert_moving_stack.py --ack-simulation-only'
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py'
echo 'PASS: moving stack'
'''
record = common + '''/usr/bin/python3 scripts/setup_camera.py
EP="${VLA_EPISODE_NAME:-demo_$(date +%Y%m%d_%H%M%S)}"
"${C[@]}" up -d --force-recreate sim moveit
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; for i in $(seq 1 45); do gz topic -l | grep -qx /station/overhead/image && exit 0; sleep 1; done; echo "No camera topic"; exit 2'
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0"; /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py'
"${C[@]}" exec -T \\
  -e "VLA_BELT_SPEED_M_S=${VLA_BELT_SPEED_M_S:-0.020}" \\
  -e "VLA_WARMUP_SIM_S=${VLA_WARMUP_SIM_S:-0}" \\
  -e "VLA_IK_SEED=${VLA_IK_SEED:-91}" \\
  moveit bash -lc "source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/record_episode.py --out /workspace/data/vla_demos/$EP --fps 5"
"${C[@]}" exec -T sim bash -lc "source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py --out /workspace/data/vla_demos/$EP/stack_verification.json"
echo "PASS: recorded episode $EP"
'''
files = {
    SCRIPTS / 'expert_moving_stack.py': s,
    SCRIPTS / 'record_episode.py': rec,
    SCRIPTS / 'run_stack.sh': launch,
    SCRIPTS / 'record_demo.sh': record,
}
if cam_old.is_file():
    files[cam_new] = cam_old.read_text()
for path, content in files.items():
    path.write_text(content)
    if path.suffix == '.sh':
        path.chmod(0o755)

report = ROOT / 'docs' / 'REFACTOR.md'
report.parent.mkdir(exist_ok=True)
report.write_text('''# Clean pipeline\n\nRun one trial: `bash scripts/run_stack.sh`.\n\nRecord one episode: `bash scripts/record_demo.sh`.\n\nCollect randomized demonstrations: `python3 scripts/collect_demos.py`.\n\nCore scripts: `expert_moving_stack.py`, `record_episode.py`, `setup_camera.py`.\n\nLegacy v7/v9 implementations remain temporarily for compatibility.\n\nEvery episode replans `00_hover` using live robot joints and the randomized green-box pose. No saved trajectory is required.\n''')
for path in files:
    print('CREATED:', path.relative_to(ROOT))
print('PASS: fresh hover IK and MoveIt planning installed; saved hover plan no longer used')
