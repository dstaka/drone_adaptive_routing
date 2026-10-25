"""Early controller flight admission, distinct from final acceptance/capture."""

import copy
import gzip
import json
from pathlib import Path
import time

from .jobs import write_json
from .payload_contract import Payload
from .payload_hybrid_controller import Controller, SENSOR_FIELDS
from .payload_hybrid_runtime import SmoothReference
from .payload_solver_mujoco import MuJoCoFixture, FREE_MASS_KG
from .payload_solver_qualification import current_solver_binding, physical_errors
from .payload_solver_runtime import SolverRuntime
from .util import sha256

_TOKEN = object()


class ReferenceAdmission:
    """Only the identical historically qualified N1 reference; never PPO admission."""

    def __init__(self, record, token):
        if token is not _TOKEN:
            raise ValueError('Rebound qualification required')
        self.record = copy.deepcopy(record)
        self.binding = copy.deepcopy(record['actual_binding'])

    @classmethod
    def from_rebinding(cls, record):
        if (record.get('schema')!='hybrid-unchanged-reference-rebinding-v1' or
            record.get('raw_manifest_sha256')!='d40d67855b46d56658eab4c097b57270cbc13b7b4d0f504c099139ac3c1d2fd1' or
            record.get('raw_files')!=2669 or record.get('raw_bytes')!=7723572007 or
            record.get('preserved_raw_bytes_identical') is not True or
            record.get('learning_admitted') is not False or
            record.get('actual_binding')!=current_solver_binding()):
            raise ValueError('Identical qualified reference and preserved raw binding required')
        return cls(record,_TOKEN)

    def require_current(self):
        if current_solver_binding()!=self.binding:
            raise ValueError('Actual reference source/build changed')


class HybridFlight:
    dt = .01

    @classmethod
    def create(cls, *, admission, original, output, case, design):
        if not isinstance(admission,ReferenceAdmission):
            raise ValueError('Development reference admission required')
        admission.require_current()
        if sha256(original)!=admission.binding['original_asset_sha256']:
            raise ValueError('Original physical asset differs')
        root=Path(output)
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/'case.json',case)
        write_json(root/'design.json',design)
        fixture=MuJoCoFixture(original,root/'fixture',Payload(FREE_MASS_KG,case['mass_ratio'],case['side_m']),substeps=16)
        if physical_errors(fixture.runtime(),dict(side_m=case['side_m'],mass_ratio=case['mass_ratio'],substeps=16),fixed=False):
            raise ValueError('Actual native case properties differ')
        write_json(root/'runtime-properties.json',fixture.runtime())
        result=cls()
        result.admission=admission
        result.runtime=SolverRuntime(fixture,root/'runtime')
        result.bank=fixture
        result.controller=Controller(design['gain'],design['hover_action'])
        result.control=gzip.open(root/'control.jsonl.gz','wt')
        result.reset_wall_s=time.monotonic()
        result.runtime.reset_case(initial=case['initial'],target=[0,0,1],target_id='S')
        result.reference=SmoothReference(result.runtime.current_sample['drone_xyz'],[0,0,1])
        result.leg_start=0.
        return result

    def sample(self):
        result=copy.deepcopy(self.runtime.current_sample)
        result.update(actions=self.runtime.last_actions.tolist(),step=self.runtime.step_count,
            time_s=self.runtime.step_count*.01,effective_target=self.runtime.target.tolist())
        return result

    def set_target(self,target,target_id=None):
        self.runtime.set_target(target,target_id=target_id)
        self.reference=SmoothReference(self.runtime.current_sample['drone_xyz'],target)
        self.leg_start=self.runtime.step_count*.01

    def advance(self):
        now=self.runtime.step_count*.01
        position,velocity=self.reference.sample(now-self.leg_start)
        sensors={k:self.runtime.current_sample[k] for k in SENSOR_FIELDS}
        action=self.controller.action(sensors,position,tick=self.runtime.step_count,reference_velocity=velocity)
        row=self.runtime.step(action)
        self.control.write(json.dumps(dict(step=row['step'],time_s=row['time_s'],
            target_id=self.runtime.target_id,target=self.runtime.target.tolist(),
            reference_position=position.tolist(),reference_velocity=velocity.tolist(),
            **copy.deepcopy(self.controller.last_audit)),separators=(',',':'),allow_nan=False)+'\n')
        return row

    def close(self):
        self.runtime.close()
        self.control.close()


def require_development_flight(flight,evidence):
    if (not isinstance(flight,HybridFlight) or
        evidence.get('hybrid_admission') is not flight.admission or
        not isinstance(flight.admission,ReferenceAdmission)):
        raise ValueError('Separate development reference capability required')
    flight.admission.require_current()
