"""Author a passive fixed-length suspension without modifying pinned assets."""

from pathlib import Path
import xml.etree.ElementTree as ET
from .payload_contract import LINK_M

LINK_MASS_KG = .0001
GIMBAL_MASS_KG = .000001


def write_payload_urdf(original, output, payload):
    original, output = Path(original), Path(output)
    if output.exists():
        raise FileExistsError(output)
    tree = ET.parse(original)
    root = tree.getroot()
    if root.find("link[@name='base_link']") is None:
        raise ValueError('Expected pinned cf2x base_link')
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((original.parent/mesh.get('filename')).resolve()))

    def link(name, mass, diagonal, center=(0, 0, 0), geometry=None):
        element = ET.SubElement(root, 'link', name=name)
        inertial = ET.SubElement(element, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, center)), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', ixx=str(diagonal[0]), iyy=str(diagonal[1]), izz=str(diagonal[2]),
                      ixy='0', ixz='0', iyz='0')
        if geometry:
            for kind in ('visual', 'collision') if name == 'payload' else ('visual',):
                geom = ET.SubElement(element, kind)
                ET.SubElement(geom, 'origin', xyz=' '.join(map(str, center)), rpy='0 0 0')
                shape = ET.SubElement(geom, 'geometry')
                ET.SubElement(shape, geometry[0], **geometry[1])
                if kind == 'visual':
                    mat = ET.SubElement(geom, 'material', name=f'{name}_material')
                    ET.SubElement(mat, 'color', rgba='0.95 0.45 0.08 1')

    def joint(name, parent, child, axis, origin):
        j = ET.SubElement(root, 'joint', name=name, type='continuous')
        ET.SubElement(j, 'parent', link=parent)
        ET.SubElement(j, 'child', link=child)
        ET.SubElement(j, 'origin', xyz=' '.join(map(str, origin)), rpy='0 0 0')
        ET.SubElement(j, 'axis', xyz=axis)
        ET.SubElement(j, 'dynamics', damping='0', friction='0')

    tiny_i = [1e-10] * 3
    link('suspension_x', GIMBAL_MASS_KG, tiny_i)
    radius = .001
    rod_i = [LINK_MASS_KG*(3*radius**2+LINK_M**2)/12]*2 + [LINK_MASS_KG*radius**2/2]
    link('suspension_rod', LINK_MASS_KG, rod_i, (0, 0, -LINK_M/2),
         ('cylinder', {'radius': str(radius), 'length': str(LINK_M)}))
    link('payload_x', GIMBAL_MASS_KG, tiny_i)
    link('payload_y', GIMBAL_MASS_KG, tiny_i)
    link('payload', payload.mass, payload.inertia, geometry=('box', {'size': ' '.join([str(payload.side_m)]*3)}))
    joint('suspension_roll', 'base_link', 'suspension_x', '1 0 0', (0, 0, -.02))
    joint('suspension_pitch', 'suspension_x', 'suspension_rod', '0 1 0', (0, 0, 0))
    joint('payload_roll', 'suspension_rod', 'payload_x', '1 0 0', (0, 0, -LINK_M))
    joint('payload_pitch', 'payload_x', 'payload_y', '0 1 0', (0, 0, 0))
    joint('payload_yaw', 'payload_y', 'payload', '0 0 1', (0, 0, -payload.side_m/2))
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return dict(schema='payload-articulation-v1', suspension_length_m=LINK_M,
                link_mass_kg=LINK_MASS_KG, gimbal_mass_kg=GIMBAL_MASS_KG, gimbal_count=3,
                joint_damping=0, default_armature=0, approximation='compression-capable rigid link, no slack',
                drone_attachment_body_m=[0, 0, -.02], package_mass_kg=payload.mass,
                package_side_m=payload.side_m, package_inertia_kg_m2=payload.inertia)
