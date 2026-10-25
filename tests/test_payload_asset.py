"""URDF authoring checks, separate from solver feasibility probes."""

import xml.etree.ElementTree as ET
from adaptive_drone_patrol.payload_asset import write_payload_urdf
from adaptive_drone_patrol.payload_contract import Payload


def test_payload_asset_keeps_rotors_and_adds_only_passive_joints(tmp_path):
    original = tmp_path/'cf2x.urdf'
    original.write_text('''<robot name="cf2"><properties kf="3.16e-10" km="7.94e-12"/>
<link name="base_link"><inertial><mass value=".027"/></inertial></link>
<link name="prop0_link"/><link name="prop1_link"/><link name="prop2_link"/><link name="prop3_link"/>
</robot>''')
    path = tmp_path/'loaded.urdf'
    result = write_payload_urdf(original, path, Payload(.027, .2, .045))
    tree = ET.parse(path).getroot()
    assert tree.find('properties').get('kf') == '3.16e-10'
    assert {x.get('name') for x in tree.findall('link')} >= {f'prop{i}_link' for i in range(4)}
    joints = tree.findall('joint')
    assert len(joints) == 5
    assert all(x.get('type') == 'continuous' for x in joints)
    assert not tree.findall('transmission')
    box = tree.find("link[@name='payload']/collision/geometry/box")
    assert box.get('size') == '0.045 0.045 0.045'
    inertial = tree.find("link[@name='payload']/inertial")
    assert float(inertial.find('mass').get('value')) == .0054
    assert abs(float(inertial.find('inertia').get('ixx')) - .0054*.045**2/6) < 1e-15
    assert result['suspension_length_m'] == .3
    assert result['joint_damping'] == 0
    assert result['link_mass_kg'] > 0
