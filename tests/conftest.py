"""Keep routing dependencies isolated from the dedicated flight environment."""

import xml.etree.ElementTree as ET
import pytest


@pytest.fixture
def minimal_cf2x_xml(tmp_path):
    """Structural writer input only; never numerical or original-asset evidence."""
    root = ET.Element("robot", name="STRUCTURAL_TEST_ONLY")
    positions = [(0, 0, 0), (0.028, -0.028, 0), (-0.028, -0.028, 0), (-0.028, 0.028, 0), (0.028, 0.028, 0)]
    for i, name in enumerate(["base_link", *[f"prop{j}_link" for j in range(4)]]):
        body = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(body, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, positions[i])))
        ET.SubElement(inertial, "mass", value=".027" if i == 0 else "0")
        diag = ["1.4e-5", "1.4e-5", "2.17e-5"] if i == 0 else ["0"] * 3
        ET.SubElement(inertial, "inertia", ixx=diag[0], iyy=diag[1], izz=diag[2], ixy="0", ixz="0", iyz="0")
        visual = ET.SubElement(body, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=name + ".obj")
        if i == 0:
            collision = ET.SubElement(body, "collision")
            ET.SubElement(collision, "origin", xyz="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", radius=".06", length=".025")
        else:
            joint = ET.SubElement(root, "joint", name=f"prop{i - 1}_joint", type="fixed")
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    path = tmp_path / "structural.urdf"
    ET.ElementTree(root).write(path)
    return path


def pytest_ignore_collect(collection_path, config):
    if config.option.markexpr.strip() == "gpu" and collection_path.name.startswith("test_"):
        return collection_path.name != "test_gpu.py"
    return None
