"""Prepare the Luxonis OAK-D Pro enclosure STL for MuJoCo (needs ``pip install trimesh fast-simplification rtree``).

    curl -LO https://oak-files.fra1.cdn.digitaloceanspaces.com/OAK-D-Pro/DM9098Pro_enclosure.STL
    python prep_mesh.py DM9098Pro_enclosure.STL oak_d_pro.stl 30000

Millimetres -> metres, origin at the bounding-box centre, decimated (quadric) to ~30k faces, vertices clipped to the
original bounding box (decimation otherwise leaves a few <=1.7 mm corner spikes). The result is 97.0 x 29.5 x 23.1 mm,
front (lens) plate on the +z side, finned heatsink on the -z side.
"""

import sys

import numpy as np
import trimesh

src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = trimesh.load(src, force="mesh")
m.apply_scale(0.001)
m.apply_translation(-m.bounds.mean(0))
lo, hi = m.bounds
d = m.simplify_quadric_decimation(face_count=target)
d.vertices = np.clip(d.vertices, lo, hi)  # decimation leaves a few <=1.7 mm corner spikes; the envelope is exact
d.update_faces(d.nondegenerate_faces())
d.remove_unreferenced_vertices()
d.export(dst)
print(
    f"faces {len(m.faces)} -> {len(d.faces)}; extents (mm) {(d.extents * 1000).round(2).tolist()} "
    f"(original {(m.extents * 1000).round(2).tolist()}); watertight {d.is_watertight}"
)
