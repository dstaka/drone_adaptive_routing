"""Elevated development flight, using the unchanged qualified observed solver."""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
from .payload_hybrid_flight import HybridFlight,ReferenceAdmission
from .payload_hybrid_duration import ObservedFixture,StageWriter
from .payload_solver_runtime import SolverRuntime
from .payload_solver_mujoco import FREE_MASS_KG
from .payload_solver_qualification import physical_errors
from .payload_contract import Payload
from .payload_hybrid_controller import Controller
from .payload_hybrid_runtime import SmoothReference
from .jobs import write_json
from .util import sha256


class ElevatedFlight(HybridFlight):
    @classmethod
    def create(cls,*,admission,original,output,case,design):
        if not isinstance(admission,ReferenceAdmission):raise ValueError('Exact qualified reference capability required')
        admission.require_current()
        if sha256(original)!=admission.binding['original_asset_sha256']:raise ValueError('Original asset differs')
        root=Path(output); root.mkdir(parents=True,exist_ok=False)
        write_json(root/'case.json',case);write_json(root/'design.json',design)
        fixture=ObservedFixture(original,root/'fixture',Payload(FREE_MASS_KG,case['mass_ratio'],case['side_m']),substeps=16)
        if physical_errors(fixture.runtime(),dict(side_m=case['side_m'],mass_ratio=case['mass_ratio'],substeps=16),fixed=False):raise ValueError('Actual model properties differ')
        write_json(root/'runtime-properties.json',fixture.runtime())
        result=cls();result.admission=admission;result.bank=fixture
        result.runtime=SolverRuntime(fixture,root/'runtime');result.controller=Controller(design['gain'],design['hover_action'])
        result.control=gzip.open(root/'control.jsonl.gz','wt')
        (root/'stages').mkdir();result.stages=StageWriter(root/'stages')
        result.reset_wall_s=time.monotonic()
        result.runtime.reset_case(initial=case['initial'],target=[0,0,1],target_id='S')
        result.reference=SmoothReference(result.runtime.current_sample['drone_xyz'],[0,0,1]);result.leg_start=0.
        return result
    def advance(self):
        from .payload_solver_work import pack_stages
        result=super().advance()
        self.stages.append(pack_stages(self.bank.stage_rows),np.asarray([r['qacc_before_solve'] for r in self.bank.stage_rows],dtype=np.float64))
        return result
    def close(self):
        self.stages.close()
        super().close()


def run_development(case,*,output,reference,original,design,routing,camera,selection=None,provisional=False):
    from .elevated_binding import development_binding,verify_binding_scenario
    from .analytical import content_hash
    from .routing_agent import RoutingAgent
    from .scenario import Scenario
    from .util import load_config
    from .payload_recovery_mission import execute_payload_mission
    from .payload_solver_acceptance import verify_mission
    from .elevated_physical import mission_vertical_report
    from .elevated_numeric_audit import audit_numeric
    binding=development_binding(routing,camera,selection=selection,provisional=provisional)
    admission=ReferenceAdmission.from_rebinding(json.loads(Path(reference).read_text()))
    scenario=Scenario.from_dict(load_config(f"configs/demo_scenarios_payload_elevated/{case['scenario']}.yaml"))
    verify_binding_scenario(scenario,case,binding)
    planner=RoutingAgent(routing,binding['routing_sha256'])
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'case.json',case);write_json(root/'elevated-binding.json',binding)
    sources={str(p):sha256(p) for p in Path('src/adaptive_drone_patrol').glob('*.py')}
    write_json(root/'executed-source.json',sources)
    for name in sources:
        saved=root/'executed-source'/name; saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(Path(name).read_bytes())
    flight=None;error=None
    try:
        flight=ElevatedFlight.create(admission=admission,original=original,output=root/'flight',case=case,design=json.loads(Path(design).read_text()))
        summary=execute_payload_mission(flight,planner,scenario,root/'mission',
            dict(evidence_kind='physical',case_id=case['case_id'],checkpoint_sha256=sha256(root/'flight/design.json'),controller='C-local-lqr-v1',
                 backend='MuJoCo3.15.0/float64/RK4/substeps16/N1',actual_binding=admission.binding,routing_model_hash=planner.hash,
                 routing_model_name=binding['routing_model_name'],flight_nn='NOT_USED',scenario=case['scenario'],
                 elevated_binding_sha256=content_hash(binding),provisional=provisional),
            physical_evidence=dict(hybrid_admission=admission),revision2=True)
    except BaseException as exc:
        error=repr(exc)
    finally:
        if flight is not None:flight.close()
        write_json(root/'final-flush.json',dict(flushed=flight is not None,error=error,
            elapsed_wall_s=time.monotonic()-flight.reset_wall_s if flight else None))
    if error:raise RuntimeError(error)
    native=dict(native_path=root/'flight/runtime/raw.jsonl.gz',properties_path=root/'flight/runtime-properties.json',
        checkpoint=root/'flight/design.json',expected_hash=sha256(root/'flight/design.json'),case=case,binding=admission.binding,
        control_path=root/'flight/control.jsonl.gz',admission=admission)
    checked=verify_mission(root/'mission',planner,native=native,revision2=True,elevated_binding=binding)
    write_json(root/'independent-mission.json',checked)
    numeric=audit_numeric(root,original=original)
    write_json(root/'independent-numeric.json',numeric)
    vertical=mission_vertical_report(root) if checked['success'] else dict(passed=False,reason='incomplete mission')
    write_json(root/'vertical.json',vertical)
    if sources!={str(p):sha256(p) for p in Path('src/adaptive_drone_patrol').glob('*.py')}:raise ValueError('Source changed during trial')
    result=dict(case_id=case['case_id'],mission=checked,numeric=numeric,vertical=vertical,
                provisional=provisional,success=checked['physical_acceptance'] and numeric['passed'],
                flush=json.loads((root/'final-flush.json').read_text()),binding_sha256=content_hash(binding))
    write_json(root/'result.json',result)
    return result


def run_vertical(case,*,output,reference,original,design,routing,camera,selection):
    from .elevated_binding import development_binding
    from .payload_hybrid_final import _record_validation
    from .payload_hybrid_final_audit import reconstruct_sequence
    from .payload_hybrid_native import verify_hybrid_native
    from .elevated_numeric_audit import audit_numeric
    binding=development_binding(routing,camera,selection=selection)
    admission=ReferenceAdmission.from_rebinding(json.loads(Path(reference).read_text()))
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'case.json',case);write_json(root/'elevated-binding.json',binding)
    flight=ElevatedFlight.create(admission=admission,original=original,output=root/'flight',case=case,design=json.loads(Path(design).read_text()))
    try:_record_validation(flight,case,root/'sequence')
    finally:
        flight.close();write_json(root/'final-flush.json',dict(flushed=True,elapsed_wall_s=time.monotonic()-flight.reset_wall_s))
    rows=[json.loads(l) for l in (root/'sequence/samples.jsonl').read_text().splitlines()]
    flush=json.loads((root/'final-flush.json').read_text())['elapsed_wall_s']
    checked=reconstruct_sequence(rows,case,flush_elapsed_wall_s=flush)
    # Inherited reader labels every first command a hold. Here it is actually
    # S->A. Independently classify using the declared starting reference S.
    seq=[[0.,0.,1.],*case['targets']]
    transfers=[a!=b for a,b in zip(seq,seq[1:])]
    if len(transfers)!=10 or not all(transfers):raise ValueError('All120 commands must be distinct-coordinate transfers')
    checked['distinct_coordinate_transfers']=sum(transfers[:checked['completed_targets']])
    checked['initial_or_duplicate_holds']=0
    for leg in checked['legs']:leg['kind']='coordinate_transfer'
    native=verify_hybrid_native(rows,native_path=root/'flight/runtime/raw.jsonl.gz',properties_path=root/'flight/runtime-properties.json',
        checkpoint=root/'flight/design.json',expected_hash=sha256(root/'flight/design.json'),case=case,binding=admission.binding,
        control_path=root/'flight/control.jsonl.gz',admission=admission)
    numeric=audit_numeric(root,original=original)
    result=dict(case_id=case['case_id'],success=checked['success'] and numeric['passed'],sequence=checked,native=native,numeric=numeric,binding=binding)
    write_json(root/'result.json',result);return result
