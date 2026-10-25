"""Independent original URDF importer for same-backend motor equivalence."""

from pathlib import Path

from .payload_contract import Payload
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG, BASE_MASS_KG, ROTOR_MASS_KG, AUTHORED_MASS_KG
from .util import sha256


def original_fixture(original, output, *, substeps=16):
    """Import the original URDF directly, retaining authored COM/body frames.

    Preserve the independently observed original Drone-importer epsilon rotor
    masses. The unused URDF center-of-mass marker remains a zero-inertia fixed
    frame. It contributes no mass, collision, degrees of freedom, or actuation.
    """
    original = Path(original)
    fixture = MuJoCoFixture(
        original, output, Payload(FREE_MASS_KG, 0.2, 0.045), loaded=False, substeps=substeps
    )
    mj = fixture.mj
    spec = mj.MjSpec.from_file(str(original))
    spec.compiler.fusestatic = False
    base = spec.body("base_link")
    base.add_freejoint()
    base.pos = [0, 0, 1]
    base.mass = BASE_MASS_KG
    for i in range(4):
        spec.body(f"prop{i}_link").mass = ROTOR_MASS_KG
    spec.option.timestep = 0.01 / substeps
    spec.option.integrator = mj.mjtIntegrator.mjINT_RK4
    spec.option.gravity = [0, 0, -9.81]
    spec.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1])
    fixture.model = spec.compile()
    fixture.model.dof_damping[3:6] = AUTHORED_MASS_KG * 1e-5
    fixture.data = mj.MjData(fixture.model)
    fixture.ids = {
        name: mj.mj_name2id(fixture.model, mj.mjtObj.mjOBJ_BODY, name)
        for name in ["base_link", *[f"prop{i}_link" for i in range(4)]]
    }
    (fixture.output / "model.xml").write_text(spec.to_xml())
    fixture.asset.update(fixture="independent-original-urdf-import", original_asset_sha256=sha256(original))
    fixture.reset()
    return fixture
