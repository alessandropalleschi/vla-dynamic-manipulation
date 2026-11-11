"""Generate the flange adapter plates (visual + hull-collision meshes) used between the UF850 flange and the VGC10.

    pip install trimesh shapely mapbox-earcut        # build-time only, not project dependencies
    python vla_mujoco/sim/assets/adapter/build_adapter.py

``adapter_disc.stl``   15 mm disc with a central bore and a 6-hole bolt circle (the camera-less tool).
``adapter_wing.stl``   the wing's wedge alone: a *convex* solid, used as its collision shape (the full plate is not
                       convex, and the convex hull of the whole plate would bridge the step and poke into the camera).
``adapter_plate.stl``  the same disc plus a wing (+y) whose upper face is an inclined plane at the angle that points
                       the OAK-D Pro at the cup tips: the camera sits directly on that face, so the face angle *is* the
                       camera tilt. The plane comes from ``scene.oak_pose()``, so mesh and camera pose cannot drift
                       apart.

All dimensions are metres in the tool frame (flange face at z = 0, +z towards the cups). The bolt circle is
illustrative: it was not matched to the xArm flange or the VGC10 interface drawings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, box

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scene", HERE.parent.parent / "scene.py")  # avoid importing mujoco
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)

FILLET = 0.010  # concave fillets where the wing meets the disc
EDGE_R = 0.006  # convex corner rounding


def outline(wing: bool):
    disc = Point(0, 0).buffer(scene.ADAPTER_R, 96)
    if not wing:
        return disc
    shape = disc.union(box(-scene.WING_HALF_W, 0.0, scene.WING_HALF_W, scene.WING_TIP_Y))
    shape = shape.buffer(FILLET, 48).buffer(-FILLET, 48)  # closing: fillets the concave corners
    return shape.buffer(-EDGE_R, 48).buffer(EDGE_R, 48)  # opening: rounds the convex corners


def with_holes(shape):
    holes = Point(0, 0).buffer(scene.ADAPTER_BORE_R, 64)
    for k in range(scene.ADAPTER_N_HOLES):
        a = np.pi / 6 + 2 * np.pi * k / scene.ADAPTER_N_HOLES
        c = scene.ADAPTER_HOLE_RING_R
        holes = holes.union(Point(c * np.cos(a), c * np.sin(a)).buffer(scene.ADAPTER_HOLE_R, 32))
    return shape.difference(holes)


def build(wing: bool) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None]:
    base = trimesh.creation.extrude_polygon(with_holes(outline(wing)), scene.ADAPTER_T)  # z in [0, T]
    if not wing:
        return base, None
    pose = scene.oak_pose()
    d, q = pose["d"], pose["plane_point"]
    # inclined plane d.(X - Q) = 0:  z(y) = Qz + (-dy/dz) (y - Qy); it meets the flange plane (z = 0) at y_start
    slope = -d[1] / d[2]
    y_start = q[1] - q[2] / slope
    region = outline(True).intersection(box(-scene.WING_HALF_W, y_start, scene.WING_HALF_W, scene.WING_TIP_Y + 0.01))
    column = trimesh.creation.extrude_polygon(region, 0.15)  # tall, then cut by the plane
    wedge = trimesh.intersections.slice_mesh_plane(
        column, plane_normal=-d, plane_origin=q, cap=True
    )  # keeps d.(X-Q) <= 0
    return trimesh.util.concatenate([base, wedge]), wedge


if __name__ == "__main__":
    for wing, name in ((False, "adapter_disc.stl"), (True, "adapter_plate.stl")):
        m, wedge = build(wing)
        m.export(HERE / name)
        print(f"{name}: {len(m.faces)} faces, bounds (m) {m.bounds.round(4).tolist()}")
        if wedge is not None:
            assert wedge.is_convex, "the wing collision shape must be convex"
            wedge.export(HERE / "adapter_wing.stl")
            print(f"adapter_wing.stl: {len(wedge.faces)} faces, convex, bounds (m) {wedge.bounds.round(4).tolist()}")
