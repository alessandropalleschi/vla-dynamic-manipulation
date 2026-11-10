"""MJCF scene generator: UF850 + VGC10-style vacuum tool + conveyor + stackable boxes.

Kinematics and joint limits come from the UFACTORY ``xarm_ros2`` description
(``uf850_default_kinematics.yaml`` / ``uf850.urdf.xacro``, BSD-3, see
``assets/uf850/LICENSE_xarm_ros2.txt``). Link masses are approximations; the arm
is gravity-compensated and driven by position actuators, so they matter little.

World frame: +z up, robot base at the origin on a table whose top is z=0.
The conveyor runs along +y at x=BELT_X with its top surface at z=BELT_TOP.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "uf850"

BELT_X = 0.62
BELT_TOP = 0.06
BELT_HALF_W = 0.15
BELT_HALF_L = 1.2
TGT_HALF = 0.04          # target ("lower") box half-size
SRC_HALF = 0.03          # source ("upper") block half-size
TOOL_TCP_Z = 0.122       # flange -> cup face along tool z (m)

# (xyz, rpy) of each joint frame relative to its parent link (uf850_default_kinematics.yaml)
JOINT_FRAMES = [
    ((0, 0, 0.364), (0, 0, 0)),
    ((0, 0, 0), (1.5708, -1.5708, 0)),
    ((0.39, 0, 0), (-3.1416, 0, -1.5708)),
    ((0.15, 0.426, 0), (-1.5708, 0, 0)),
    ((0, 0, 0), (-1.5708, 0, 0)),
    ((0, -0.09, 0), (1.5708, 0, 0)),
]
JOINT_LIMITS = [
    (-math.pi * 0.99, math.pi * 0.99),
    (-2.3038346, 2.3038346),
    (-math.pi * 0.99, 0.061087),
    (-math.pi * 0.99, math.pi * 0.99),
    (-2.1642, 2.1642),
    (-math.pi * 0.99, math.pi * 0.99),
]
EFFORT = [382.5, 382.5, 243.5, 117.3, 38.25, 38.25]
LINK_MASS = [2.27, 2.0, 2.6, 1.7, 1.3, 1.1, 0.4]  # base, link1..link6 (approximate)
KP = [900, 900, 700, 250, 120, 80]
KV = [90, 90, 70, 25, 12, 8]


def rpy_to_quat(r: float, p: float, y: float) -> np.ndarray:
    """URDF rpy (fixed-axis x, y, z) -> MuJoCo quaternion (w, x, y, z)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                              math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def build_mjcf(img_size: int = 128) -> str:
    meshes = "\n    ".join(
        f'<mesh name="{n}" file="{n}.stl"/>'
        for n in ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"])

    # nested arm bodies
    arm = ""
    indent = "      "
    closers = []
    for i, ((xyz, rpy), (lo, hi)) in enumerate(zip(JOINT_FRAMES, JOINT_LIMITS), start=1):
        q = rpy_to_quat(*rpy)
        arm += (f'{indent}<body name="link{i}" pos="{_f(xyz)}" quat="{_f(q)}" gravcomp="1">\n'
                f'{indent}  <joint name="joint{i}" axis="0 0 1" range="{lo:.6g} {hi:.6g}" '
                f'damping="2" armature="0.05"/>\n'
                f'{indent}  <geom type="mesh" mesh="link{i}" mass="{LINK_MASS[i]}" class="arm"/>\n')
        closers.append(f"{indent}</body>\n")
        indent += "  "
    # tool (VGC10-like): mount plate, body, 4 suction cups (cups are non-colliding)
    tool = f"""{indent}<body name="tool" gravcomp="1">
{indent}  <geom type="cylinder" size="0.04 0.005" pos="0 0 0.005" mass="0.1" class="toolcol"/>
{indent}  <geom type="box" size="0.0505 0.05 0.05" pos="0 0 0.06" mass="0.7" class="toolcol"/>
{indent}  <geom type="cylinder" size="0.015 0.006" pos="0.025 0.025 0.116" class="cup"/>
{indent}  <geom type="cylinder" size="0.015 0.006" pos="-0.025 0.025 0.116" class="cup"/>
{indent}  <geom type="cylinder" size="0.015 0.006" pos="0.025 -0.025 0.116" class="cup"/>
{indent}  <geom type="cylinder" size="0.015 0.006" pos="-0.025 -0.025 0.116" class="cup"/>
{indent}  <site name="tcp" pos="0 0 {TOOL_TCP_Z}" size="0.004" rgba="1 0 0 1"/>
{indent}  <camera name="wrist" pos="0.0 0.075 0.09" quat="0 1 0 0" fovy="85"/>
{indent}</body>
"""
    arm += tool + "".join(reversed(closers))

    act = "\n    ".join(
        f'<position name="act{i + 1}" joint="joint{i + 1}" kp="{KP[i]}" kv="{KV[i]}" '
        f'ctrlrange="{JOINT_LIMITS[i][0]:.6g} {JOINT_LIMITS[i][1]:.6g}" '
        f'forcerange="-{EFFORT[i]} {EFFORT[i]}"/>' for i in range(6))

    box_geom = lambda half, rgba, mass: (  # noqa: E731
        f'<geom type="box" size="{half} {half} {half}" mass="{mass}" rgba="{rgba}" '
        f'class="obj" friction="1.0 0.01 0.001"/>')

    return f"""<mujoco model="uf850_dynamic_stack">
  <compiler angle="radian" meshdir="{ASSET_DIR}" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{max(img_size, 640)}" offheight="{max(img_size, 480)}"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    {meshes}
    <texture name="floor" type="2d" builtin="checker" rgb1="0.78 0.78 0.80" rgb2="0.70 0.70 0.73" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="6 6" texuniform="true" reflectance="0.0"/>
    <texture name="beltTex" type="2d" builtin="checker" rgb1="0.16 0.16 0.18" rgb2="0.30 0.30 0.33" width="64" height="64"/>
    <material name="belt" texture="beltTex" texrepeat="1 24" texuniform="false"/>
  </asset>
  <default>
    <default class="arm"><geom contype="1" conaffinity="2" rgba="0.88 0.88 0.9 1"/></default>
    <default class="toolcol"><geom contype="1" conaffinity="2" rgba="0.25 0.27 0.3 1"/></default>
    <default class="cup"><geom contype="0" conaffinity="0" rgba="0.05 0.05 0.05 1" mass="0.005"/></default>
    <default class="obj"><geom contype="2" conaffinity="3" condim="3"/></default>
  </default>
  <worldbody>
    <light pos="0.5 0 2.5" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light pos="1.5 1.0 1.5" dir="-1 -0.6 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.05" material="floor" contype="0" conaffinity="0"/>
    <geom name="table" type="box" size="0.8 0.85 0.025" pos="0.45 0 -0.025" rgba="0.62 0.55 0.45 1" class="obj"/>
    <!-- conveyor frame (static) and moving belt slab -->
    <geom name="belt_frame" type="box" size="{BELT_HALF_W + 0.012} {BELT_HALF_L} {(BELT_TOP - 0.012) / 2}" pos="{BELT_X} 0 {(BELT_TOP - 0.012) / 2}" rgba="0.35 0.37 0.4 1" contype="0" conaffinity="0"/>
    <geom name="rail_l" type="box" size="0.008 {BELT_HALF_L} 0.012" pos="{BELT_X - BELT_HALF_W - 0.009} 0 {BELT_TOP + 0.006}" rgba="0.8 0.6 0.1 1" class="obj"/>
    <geom name="rail_r" type="box" size="0.008 {BELT_HALF_L} 0.012" pos="{BELT_X + BELT_HALF_W + 0.009} 0 {BELT_TOP + 0.006}" rgba="0.8 0.6 0.1 1" class="obj"/>
    <body name="belt" pos="{BELT_X} 0 {BELT_TOP - 0.006}">
      <joint name="belt_slide" type="slide" axis="0 1 0" damping="0"/>
      <geom name="belt_top" type="box" size="{BELT_HALF_W} {BELT_HALF_L} 0.006" mass="30" material="belt" class="obj" friction="1.2 0.01 0.001"/>
    </body>
    <!-- robot -->
    <body name="base" pos="0 0 0.003">
      <geom type="mesh" mesh="link_base" mass="{LINK_MASS[0]}" class="arm"/>
{arm}    </body>
    <!-- manipulable objects -->
    <body name="src" pos="0.3 0.3 {SRC_HALF + 0.001}">
      <freejoint name="src_free"/>
      {box_geom(SRC_HALF, "0.85 0.15 0.15 1", 0.12)}
    </body>
    <body name="tgt" pos="{BELT_X} -0.4 {BELT_TOP + TGT_HALF + 0.001}">
      <freejoint name="tgt_free"/>
      {box_geom(TGT_HALF, "0.15 0.3 0.85 1", 0.3)}
    </body>
    <body name="dist" pos="{BELT_X} -0.7 {BELT_TOP + TGT_HALF + 0.001}">
      <freejoint name="dist_free"/>
      {box_geom(TGT_HALF, "0.15 0.7 0.25 1", 0.3)}
    </body>
    <!-- occluder panel (parked below the floor until used) -->
    <body name="occluder" mocap="true" pos="0.95 0 -2">
      <geom type="box" size="0.01 0.14 0.2" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    </body>
    <body name="focus" pos="0.55 0 0.1"/>
    <camera name="front" pos="1.25 0 0.62" mode="targetbody" target="focus" fovy="50"/>
    <camera name="overhead" pos="{BELT_X - 0.1} 0 1.25" xyaxes="0 -1 0 1 0 0" fovy="55"/>
  </worldbody>
  <equality>
    <weld name="grasp_src" body1="tool" body2="src" active="false"/>
    <weld name="grasp_tgt" body1="tool" body2="tgt" active="false"/>
    <weld name="grasp_dist" body1="tool" body2="dist" active="false"/>
  </equality>
  <actuator>
    {act}
    <velocity name="belt_drive" joint="belt_slide" kv="3000" ctrlrange="-0.5 0.5"/>
  </actuator>
</mujoco>
"""
