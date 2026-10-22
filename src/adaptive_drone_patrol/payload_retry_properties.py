"""Read-only effective-property checks at each retry episode boundary."""
import math
import numpy as np
from .payload_contract import rotation
from .util import sha256


def read_reset_properties(bank):
    entity, package = bank.drone, bank.package
    masses=entity.get_links_mass()[0].detach().cpu().tolist()
    index=package.idx-entity.link_start
    original=('base_link','prop0_link','prop1_link','prop2_link','prop3_link')
    fixture=bank.fixtures[bank.payload.side_m]
    asset=bank.output/f'load_{bank.payload.side_m:.3f}'/'candidate.xml'
    result=dict(original_mass_kg=sum(masses[entity.get_link(name).idx-entity.link_start] for name in original),
        package_mass_kg=masses[index],
        package_inertia_kg_m2=entity.get_links_inertia()[0,index].detach().cpu().tolist(),
        package_aabb=package.get_AABB()[0].detach().cpu().tolist(),
        qpos=entity.get_qpos()[0].detach().cpu().tolist(),
        velocity=entity.get_dofs_velocity()[0].detach().cpu().tolist(),
        asset_sha256=fixture['asset_manifest']['generated_asset_sha256'],
        asset_bytes_verified=asset.is_file() and sha256(asset)==fixture['asset_manifest']['generated_asset_sha256'])
    for body,obj in [('drone',entity.get_link('base_link')),('package',package)]:
        result[body+'_xyz']=obj.get_pos()[0].detach().cpu().tolist()
        result[body+'_quaternion_wxyz']=obj.get_quat()[0].detach().cpu().tolist()
    return result


def validate_reset_properties(record,payload,initial=None):
    errors=[]
    try:
        if (not math.isclose(record['original_mass_kg'],payload.reference_mass_kg,rel_tol=1e-5) or
                not math.isclose(record['package_mass_kg'],payload.mass,rel_tol=1e-5)):
            errors.append('EFFECTIVE_MASS_MISMATCH')
        inertia=np.asarray(record['package_inertia_kg_m2'],float)
        if inertia.shape!=(3,3) or not np.allclose(inertia,np.eye(3)*payload.inertia[0],rtol=1e-5,atol=1e-15):
            errors.append('EFFECTIVE_INERTIA_MISMATCH')
        velocity=np.asarray(record['velocity'],float)
        qpos=np.asarray(record['qpos'],float)
        if (velocity.shape!=(12,) or qpos.shape!=(15,) or not np.isfinite(velocity).all() or
                not np.isfinite(qpos).all() or np.any(velocity!=0)):
            errors.append('RESET_GENERALIZED_STATE_MISMATCH')
        if qpos.shape==(15,) and any(abs(np.linalg.norm(qpos[i:i+4])-1)>1e-5 for i in (3,7,11)):
            errors.append('RESET_QUATERNION_MISMATCH')
        rd,rp=[rotation(record[body+'_quaternion_wxyz']) for body in ('drone','package')]
        drone,package=[np.asarray(record[body+'_xyz'],float) for body in ('drone','package')]
        if drone.shape!=(3,) or package.shape!=(3,) or not np.isfinite(drone).all() or not np.isfinite(package).all():
            raise ValueError('Invalid reset position')
        attachment=package+rp @ [0,0,payload.side_m/2]-(drone+rd @ [0,0,-.02])
        if abs(np.linalg.norm(attachment)-.3)>2e-5:
            errors.append('RESET_ATTACHMENT_MISMATCH')
        extent=np.abs(rp) @ np.full(3,payload.side_m/2)
        aabb=np.asarray(record['package_aabb'],float)
        if aabb.shape!=(2,3) or not np.allclose(aabb,[package-extent,package+extent],atol=1e-5,rtol=0):
            errors.append('EFFECTIVE_COLLISION_MISMATCH')
        if record.get('asset_bytes_verified') is not True or len(record.get('asset_sha256',''))!=64:
            errors.append('RESET_ASSET_MISMATCH')
        if initial is not None:
            for body in ('drone','package'):
                if (not np.allclose(record[body+'_xyz'],initial[body+'_xyz'],atol=1e-6,rtol=0) or
                        not np.allclose(rotation(record[body+'_quaternion_wxyz']),
                                        rotation(initial[body+'_quaternion_wxyz']),atol=1e-6,rtol=0)):
                    errors.append('RESET_INITIAL_POSE_MISMATCH')
    except (KeyError,ValueError,TypeError,IndexError):
        errors.append('RESET_PROPERTY_EVIDENCE_INCOMPLETE')
    return errors
