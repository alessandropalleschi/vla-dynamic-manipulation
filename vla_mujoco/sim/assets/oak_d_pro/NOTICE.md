# OAK-D Pro enclosure mesh

`oak_d_pro.stl` is derived from the OAK-D Pro enclosure model published by Luxonis in
<https://github.com/luxonis/depthai-hardware> (folder `DM9098_OAK-D-Pro`, file linked from `Mechanical/README.md`:
`https://oak-files.fra1.cdn.digitaloceanspaces.com/OAK-D-Pro/DM9098Pro_enclosure.STL`).

* Licence of the repository: MIT, Copyright (c) 2020 luxonis, see `LICENSE_luxonis_depthai-hardware.txt`.
  The STL itself is hosted on Luxonis' CDN and linked from that repository; we assume the repository licence covers it -
  confirm with Luxonis before redistributing outside this project.
* Changes: scaled mm -> m, re-centred, decimated from 246,306 to ~30,000 triangles, vertices clipped to the original
  bounding box (`prep_mesh.py`). Used as a **visual** mesh only; collision uses a box of the true 97.0 x 29.5 x 23.1 mm size.
* The model is the enclosure only: lens openings are not modelled (the simulator adds marker discs), and the real camera's
  mounting holes were not verified, so the bracket in `vla_mujoco/sim/scene.py` is a plausible design, not a drawing.
