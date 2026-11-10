from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

def backup(p):
    dest = Path(str(p) + '.before_vgc10_flip')
    if not dest.exists(): shutil.copy2(p, dest)

def replace(p, old, new):
    text = p.read_text()
    if new in text: return
    if text.count(old) != 1: raise RuntimeError(f'Unexpected source in {p}: {old}')
    backup(p); p.write_text(text.replace(old, new, 1))

# Future regenerations must use the new mount rotation.
replace(Path('scripts/prepare_uf850_model.py'),
    'augment(raw_urdf, output, flange=flange, cups_mm=cup_mm,',
    "augment(raw_urdf, output, flange=flange, mount_rpy='3.141592653589793 0 0', cups_mm=cup_mm,")

# The URDF currently used for spawning.
p = Path('simulation/generated/uf850_vgc10.urdf')
t = ET.parse(p); j = t.getroot().find("./joint[@name='uf850_to_vgc10']")
if j is None: raise RuntimeError('Missing uf850_to_vgc10')
o = j.find('origin')
if o.get('rpy', '0 0 0') == '0 0 0':
    backup(p); o.set('rpy', '3.141592653589793 0 0'); t.write(p, encoding='utf-8', xml_declaration=True)
elif o.get('rpy') != '3.141592653589793 0 0': raise RuntimeError('Unexpected existing mount rotation')

# Keep the Gazebo plugin aligned with the transformed tool geometry.
p = Path('simulation/worlds/conveyor_station_uf850.sdf')
t = ET.parse(p); w = t.getroot().find('world')
g = w.find("./plugin[@name='conveyor::Vgc10Vacuum']")
if g is None: raise RuntimeError('Missing VGC10 plugin in world')
o = g.find('tool_offset')
if o is None or o.text.strip() not in ('0 0 -0.097', '0 0 0.127'):
    raise RuntimeError('Unexpected tool_offset')
a = g.find('suction_axis')
if a is None or o.text.strip() != '0 0 0.127':
    backup(p); o.text = '0 0 0.127'
    if a is None: a = ET.SubElement(g, 'suction_axis')
    a.text = '0 0 1'; t.write(p, encoding='utf-8', xml_declaration=True)

# Keep the default axis unchanged for the independent fixture test.
p = Path('simulation/plugins/vgc10/Vgc10Vacuum.cc')
replace(p,
    'const auto normal = flange.Rot().RotateVector(gz::math::Vector3d(0, 0, -1));',
    'const auto normal = flange.Rot().RotateVector(this->suctionAxis);')
replace(p,
    'gz::math::Vector3d toolOffset{0, 0, -0.097};',
    'gz::math::Vector3d toolOffset{0, 0, -0.097};\n  gz::math::Vector3d suctionAxis{0, 0, -1};')
replace(p,
    'this->toolOffset = _sdf->Get<gz::math::Vector3d>("tool_offset", gz::math::Vector3d(0, 0, -0.097)).first;',
    'this->toolOffset = _sdf->Get<gz::math::Vector3d>("tool_offset", gz::math::Vector3d(0, 0, -0.097)).first;\n    this->suctionAxis = _sdf->Get<gz::math::Vector3d>("suction_axis", gz::math::Vector3d(0, 0, -1)).first;')

# The installer should reproduce the corrected world settings.
p = Path('scripts/install_uf850_vacuum.py')
replace(p, "'tool_offset':'0 0 -0.097', 'cup_pitch_x'",
           "'tool_offset':'0 0 0.127', 'suction_axis':'0 0 1', 'cup_pitch_x'")

# Update the read-only compact IK probe to use the corrected TCP and axis.
p = Path('scripts/uf850_pickup_ik_probe.py')
if p.exists():
    text=p.read_text()
    if 'default=[0,0,-.097]' in text:
        replace(p, 'default=[0,0,-.097]', 'default=[0,0,.127]')
    if 'np.array([0.,0.,-1.])' in text:
        replace(p, 'np.array([0.,0.,-1.])', 'np.array([0.,0.,1.])')
    elif 'normal = T[:3,:3] @ np.array([0.0,0.0,-1.0])' in text:
        replace(p, 'normal = T[:3,:3] @ np.array([0.0,0.0,-1.0])',
                   'normal = T[:3,:3] @ np.array([0.0,0.0,1.0])')

print('VGC10 mount, Gazebo suction geometry, and IK defaults corrected.')
