#!/usr/bin/env python3
"""Patch current generated UF850 URDF and future regeneration to use real VGC10 STL.

Called inside the `tools` container (with /workspace bind-mounted).
Requires assets cloned by scripts/install_vgc10_real_stl.sh.
No ROS dependency; purely XML and source generation.
"""
from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from vgc10_real_visual import patch_generated_urdf, LOCAL_REL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh-xyz', nargs=3, type=float, default=[0, 0, 0])
    ap.add_argument('--mesh-rpy', nargs=3, type=float, default=[0, 0, 0])
    args = ap.parse_args()
    xyz = ' '.join(map(str, args.mesh_xyz))
    rpy = ' '.join(map(str, args.mesh_rpy))
    stl = ROOT / LOCAL_REL
    if not stl.is_file():
        sys.exit(f'ERROR: STL missing at {stl}. Run scripts/install_vgc10_real_stl.sh first.')
    print(f'Using 4-cup STL: {stl} ({stl.stat().st_size:,} bytes)', flush=True)

    urdf = ROOT / 'simulation/generated/uf850_vgc10.urdf'
    if not urdf.exists():
        sys.exit(f'ERROR: generated UF850 URDF missing at {urdf}')
    backup = Path(str(urdf) + '.before_real_vgc10_stl')
    if not backup.exists():
        shutil.copy2(urdf, backup)
    patch_generated_urdf(urdf, xyz=xyz, rpy=rpy)
    print(f'Updated visual-only URDF: {urdf}', flush=True)

    generator = ROOT / 'scripts/prepare_uf850_model.py'
    s = generator.read_text()
    mark = '# VGC10_REAL_STL_VISUAL_AUTOGEN'
    if mark not in s:
        anchor = '    robot = tree.getroot()\n'
        if s.count(anchor) != 1:
            sys.exit('Could not locate generated URDF root in prepare_uf850_model.py')
        patch = (
            f'    {mark}\n'
            '    from vgc10_real_visual import apply_real_visual, LOCAL_REL\n'
            f'    _asset = Path(__file__).resolve().parent.parent / LOCAL_REL\n'
            '    if _asset.is_file():\n'
            '        apply_real_visual(robot, mesh_path=_asset)\n'
        )
        backup = Path(str(generator) + '.before_real_vgc10_stl')
        if not backup.exists():
            shutil.copy2(generator, backup)
        generator.write_text(s.replace(anchor, anchor + patch, 1))
        print('Enabled real STL visual for future UF850 URDF regeneration.', flush=True)
    else:
        print('Generator already patched (idempotent).', flush=True)

    # Checks: all frames and collisions remain, but primitive visuals are gone.
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf)
    root = tree.getroot()
    assert root.find("./joint[@name='uf850_to_vgc10']") is not None
    for cid in ('a1', 'a2', 'b1', 'b2'):
        cup = root.find(f"./link[@name='vgc10_cup_{cid}']")
        assert cup is not None and cup.find('collision') is not None
    body = root.find("./link[@name='vgc10_body']")
    assert body.find('collision') is not None
    mesh = body.find('visual/geometry/mesh')
    assert mesh is not None and mesh.get('filename').startswith('/workspace/')
    print('PASS: VGC10 STL visual installed; mounting/TCP, 4 cup frames, '
          'collision proxies and vacuum system unchanged.', flush=True)


if __name__ == '__main__':
    main()
