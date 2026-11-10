# VGC10 four-cup STL visual integration

Source: https://github.com/UOsaka-Harada-Laboratory/onrobot (MIT license, original source package includes license). Four-cup visual STL path: `onrobot_vg_description/meshes/vgc10/visual/base_link_4cups.stl`. The ROS xacro declares that mesh with identity visual origin and no scaling. This project fetches the mesh from its published repository on the user's machine; the ZIP does **not** bundle any manufacturer's CAD assets.

From `~/conveyor-belief-vla` after extraction:

```bash
bash scripts/install_vgc10_real_stl.sh

docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml restart sim
```

Then capture `uf850_side.png` again. The STL replaces **only** the VGC10 body-and-four-cups visual. The four cup links, collision proxies, inertial parameters, vacuum attachment plugin, tool mounting transform and TCP are untouched. This is intentional: CAD-triangle collision meshes are often unstable or slow in physics and we have not calibrated this visual STL's geometry against the user's physical four-cup arrangement.

**Note:** The STL's internal coordinate origin may differ from the current surrogate `vgc10_body` geometry. The initial mesh origin is identity, as in the upstream xacro. Inspect its placement in the side camera. Adjust the mesh visual origin separately if necessary; do **not** change the physical mount, TCP or vacuum plugin just to improve appearance.

To tweak only visual origin (example, 10 mm local Z correction):

```bash
bash scripts/install_vgc10_real_stl.sh --mesh-xyz 0 0 0.01
```

The custom visual adjustment currently applies to the generated URDF. If you regenerate the URDF, the installed hook defaults to the upstream identity visual pose, so rerun the adjustment after regeneration.

Validate generated URDF mesh reference:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml exec tools bash -c \
  'grep -A4 vgc10_manufacturer_style_4cup_stl /workspace/simulation/generated/uf850_vgc10.urdf'
```

If the visual model has an unexpected orientation despite being aligned in the published xacro, calibrate the *mesh visual frame* before reconsidering the tool mount or suction orientation.
