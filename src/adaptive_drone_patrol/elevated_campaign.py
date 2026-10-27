"""Typed joint freeze and separate fresh routing/physical held-out cohorts."""
import copy,gzip,json
from pathlib import Path
import numpy as np
from .analytical import content_hash
from .util import sha256,json_write
from .payload_cases import initial_state,SCENARIOS
from .elevated_routing_experiment import select_model,BASELINE_SHA256
from .elevated_binding import development_binding
from .elevated_physical import vertical_cases,mission_vertical_report
from .elevated_artifacts import inventory,verify_inventory

_TOKEN=object()
PHYSICAL_ROOT=10262501
# These dependencies of the already executed flight/audit must remain identical.
# New orchestration/media/report modules may be added before joint freeze; that
# does not rebind a historical physical trace to modified physical logic.
DEPENDENCIES=['elevated_flight','elevated_numeric_audit','elevated_binding','elevated_layout','elevated_physical',
 'payload_hybrid_flight','payload_hybrid_controller','payload_hybrid_runtime','payload_hybrid_contract','payload_hybrid_native',
 'payload_hybrid_audit','payload_recovery_mission','payload_solver_acceptance','payload_solver_validation','payload_solver_trace',
 'payload_solver_mujoco','payload_solver_runtime','payload_solver_work','payload_solver_work_check','payload_solver_initial_reset',
 'payload_contract','payload_cases','payload_ball_asset','payload_ball_bank','payload_solver_engine','scenario','objective',
 'routing_agent','routing_policy','analytical','baselines','physical_evidence']


def development_cases():
 from .payload_hybrid_final import robustness_cases
 cases=robustness_cases()
 for c in cases:c['case_id']=c['case_id'].replace('final-development','elevated-development');c['suite']='elevated-development'
 return cases


def audit_source_hashes():
 return {str(p):sha256(p) for p in sorted(Path('src/adaptive_drone_patrol').glob('*.py'))}


def require_same_controller(actual,declared):
 # The flight writer serializes the already declared values in its own JSON
 # formatting. Freeze still binds both files' exact bytes independently.
 if json.loads(Path(actual).read_text())!=json.loads(Path(declared).read_text()):raise ValueError('Controller differs')


def validate_audit_receipt(trial,case,receipt):
 """Reject stale independent checks after any consumed evidence/source changes."""
 if receipt['case_id']!=case['case_id'] or receipt['checked_sources']!=audit_source_hashes():raise ValueError('Independent audit case/source differs')
 verify_inventory(trial,receipt['artifacts'])
 if inventory(trial)!=receipt['artifacts']:raise ValueError('Independent audit input inventory changed')
 return receipt


def audit_prerequisite(trial,case,paths,vertical=False):
 from .payload_hybrid_flight import ReferenceAdmission
 from .payload_solver_acceptance import verify_mission
 from .payload_hybrid_final_audit import reconstruct_sequence
 from .payload_hybrid_native import verify_hybrid_native
 from .elevated_numeric_audit import audit_numeric
 from .routing_agent import RoutingAgent
 root=Path(trial)
 before=inventory(root);sources=audit_source_hashes()
 if json.loads((root/'case.json').read_text())!=case:raise ValueError('Prerequisite allocation changed')
 old=json.loads((root/'executed-source.json').read_text())
 for module in DEPENDENCIES:
  name=f'src/adaptive_drone_patrol/{module}.py'
  if name not in old or sha256(name)!=old[name]:raise ValueError('Relevant prerequisite source changed: '+name)
 binding=development_binding(paths['routing'],paths['camera'],selection=paths['selection'])
 if json.loads((root/'elevated-binding.json').read_text())!=binding:raise ValueError('Mixed selected model/layout/camera')
 admission=ReferenceAdmission.from_rebinding(json.loads(Path(paths['reference']).read_text()))
 native=dict(native_path=root/'flight/runtime/raw.jsonl.gz',properties_path=root/'flight/runtime-properties.json',
   checkpoint=root/'flight/design.json',expected_hash=sha256(root/'flight/design.json'),case=case,binding=admission.binding,
   control_path=root/'flight/control.jsonl.gz',admission=admission)
 require_same_controller(root/'flight/design.json',paths['design'])
 if vertical:
  rows=[json.loads(l) for l in (root/'sequence/samples.jsonl').read_text().splitlines()]
  flush=json.loads((root/'final-flush.json').read_text())['elapsed_wall_s']
  checked=reconstruct_sequence(rows,case,flush_elapsed_wall_s=flush)
  verify_hybrid_native(rows,**native)
  sequence=[[0.,0.,1.],*case['targets']]
  if not all(a!=b for a,b in zip(sequence,sequence[1:])) or checked['completed_targets']!=10:raise ValueError('120 actual distinct transfers required')
  checked['distinct_coordinate_transfers']=10;checked['initial_or_duplicate_holds']=0
 else:
  planner=RoutingAgent(paths['routing'],binding['routing_sha256'])
  checked=verify_mission(root/'mission',planner,native=native,revision2=True,elevated_binding=binding)
  if case.get('allocation')=='nominal' and not mission_vertical_report(root)['passed']:raise ValueError('Selected demo route cannot meet vertical display')
 numeric=audit_numeric(root,original=paths['original'])
 if not checked['success'] or not numeric['passed']:raise ValueError('New-height prerequisite gate failed')
 receipt=dict(case_id=case['case_id'],passed=True,checked=checked,numeric=numeric,artifacts=before,checked_sources=sources)
 validate_audit_receipt(root,case,receipt)
 return receipt


class JointFreeze:
 def __init__(self,record,token):
  if token is not _TOKEN:raise ValueError('Independently audited joint freeze required')
  from .payload_hybrid_final_audit import VerifiedArtifactSet
  self.record=copy.deepcopy(record);self._files=VerifiedArtifactSet(record['files']);self._formal=None
 def require_current(self):
  self._files.require_current()
  import importlib.metadata as md
  if any(md.version(k)!=v for k,v in self.record['dependencies'].items()):raise ValueError('Frozen dependency versions changed')
 @classmethod
 def create(cls,run,*,development,vertical,paths,framing,output=None):
  from .payload_hybrid_flight import ReferenceAdmission
  from .payload_hybrid_final_audit import VerifiedArtifactSet
  run,development,vertical,framing=map(Path,(run,development,vertical,framing))
  binding=development_binding(paths['routing'],paths['camera'],selection=paths['selection'])
  ReferenceAdmission.from_rebinding(json.loads(Path(paths['reference']).read_text())).require_current()
  selection=json.loads(Path(paths['selection']).read_text());prospective=select_model(selection['scores'])
  if prospective['selected']!=selection['selected'] or selection['selected'] is None or set(selection['models'])!={'B','F42','F43','F44'}:raise ValueError('Invalid frozen routing selection')
  if selection['models']['B']['sha256']!=BASELINE_SHA256:raise ValueError('Baseline changed')
  for name in ('F42','F43','F44'):
   training=json.loads((run/name/'training.json').read_text())
   if (training['static_actual'],training['mixture_actual'],training['total_actual'])!=(50176,50176,100352):raise ValueError('Finite three-run comparison incomplete')
   if sha256(run/name/'terminal.zip')!=selection['models'][name]['sha256']:raise ValueError('Frozen terminal model differs')
  portability=json.loads((run/'inference-portability.json').read_text())
  if not portability['passed'] or len(portability['rows'])!=1200 or any(not r['order_equal'] or r['cost_error']>1e-10 for r in portability['rows']):raise ValueError('Inference dependency compatibility not verified')
  if json.loads((development/'declaration.json').read_text())['cases']!=development_cases() or json.loads((vertical/'declaration.json').read_text())['cases']!=vertical_cases():raise ValueError('Development/vertical declaration changed')
  # Independent source-bound raw re-audits have just been performed by the
  # freeze preparation command. Bind all actual raw/stage/audit bytes below.
  for directory,cases in [(development,development_cases()),(vertical,vertical_cases())]:
   for case in cases:
    audit_path=run/'joint-prerequisite-audits'/f"{case['case_id']}.json"
    audit=json.loads(audit_path.read_text())
    if not audit['passed'] or audit['case_id']!=case['case_id']:raise ValueError('Incomplete independent prerequisite audit')
    validate_audit_receipt(directory/case['case_id'],case,audit)
    if audit['numeric']['raw_sha256']!=sha256(directory/case['case_id']/'flight/runtime/raw.jsonl.gz'):raise ValueError('Stale numerical audit')
    if audit['numeric']['auditor_sha256']!=sha256('src/adaptive_drone_patrol/elevated_numeric_audit.py'):raise ValueError('Numeric auditor source differs')
    if audit['numeric']['stage_index_sha256']!=sha256(directory/case['case_id']/'flight/stages/stages-index.json'):raise ValueError('Stage evidence differs')
    index=json.loads((directory/case['case_id']/'flight/stages/stages-index.json').read_text())
    for chunk in index:
     if sha256(directory/case['case_id']/'flight/stages'/chunk['path'])!=chunk['sha256']:raise ValueError('Stage chunk changed')
  for case in development_cases():
   proof=json.loads((framing/f"{case['case_id']}.json").read_text())
   if proof['raw_sha256']!=sha256(development/case['case_id']/'flight/runtime/raw.jsonl.gz') or proof['camera_sha256']!=binding['camera_sha256'] or proof['median_width_ratio']<1.2:raise ValueError('Full selected-development camera qualification missing')
  files={}
  roots=[development,vertical,framing,run/'joint-prerequisite-audits',run/'routing-validation',run/'routing-inputs']
  for root in roots:
   for p in root.rglob('*'):
    if p.is_file():files[str(p)]=sha256(p)
  for d in ('src','scripts','configs','requirements','tests'):
   for p in Path(d).rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store':files[str(p)]=sha256(p)
  for name in ('B','F42','F43','F44'):
   p=run/'baseline/B.zip' if name=='B' else run/name/'terminal.zip'
   for q in (p,p.with_suffix('.schema.json')):files[str(q)]=sha256(q)
  for p in [*map(Path,paths.values()),run/'routing-declaration.json',run/'validation-manifest.json',run/'inference-portability.json',run/'frozen-routing-input-hashes.json']:
   files[str(p)]=sha256(p)
  import importlib.metadata as md
  engine=md.distribution('mujoco')
  from .payload_solver_engine import is_engine_file
  for item in engine.files:
   p=Path(engine.locate_file(item))
   if is_engine_file(item) and p.is_file():files[str(p)]=sha256(p)
  font=Path(paths['original']).parent.parent/'NotoSansJP.ttf'
  files[str(font)]=sha256(font)
  for p in Path(paths['original']).parent.rglob('*'):
   if p.is_file():files[str(p)]=sha256(p)
  dependencies={k:md.version(k) for k in ['numpy','mujoco','torch','gymnasium','stable-baselines3','sb3-contrib','PyYAML']}
  record=dict(dependencies=dependencies,schema='payload-elevated-joint-freeze-v1',binding=binding,paths={k:str(v) for k,v in paths.items()},
   selected=selection['selected'],selected_sha256=binding['routing_sha256'],routing_models=selection['models'],run=str(run),
   development=str(development),vertical=str(vertical),framing=str(framing),files=files,
   routing_root=10262502,physical_root=PHYSICAL_ROOT,routing_heldout='UNOPENED_AT_FREEZE',physical_heldout='UNOPENED_AT_FREEZE')
  obj=cls(record,_TOKEN)
  if output is not None:
   if Path(output).exists():raise FileExistsError(output)
   json_write(output,record)
  return obj
 @classmethod
 def load(cls,path):
  record=json.loads(Path(path).read_text())
  if record.get('schema')!='payload-elevated-joint-freeze-v1' or record['routing_heldout']!='UNOPENED_AT_FREEZE' or record['physical_heldout']!='UNOPENED_AT_FREEZE':raise ValueError('Invalid joint freeze')
  current=cls.create(record['run'],development=record['development'],vertical=record['vertical'],paths=record['paths'],framing=record['framing'])
  if current.record!=record:raise ValueError('Joint freeze differs from complete current prerequisite/source/model binding')
  return current


def formal_cases(freeze):
 if not isinstance(freeze,JointFreeze):raise ValueError('Typed joint freeze required before physical held-out')
 freeze.require_current()
 if freeze._formal is not None:return copy.deepcopy(freeze._formal)
 cases=[];root=PHYSICAL_ROOT
 for scenario in SCENARIOS:
  for seed in (1,2,3):
   rng=np.random.default_rng(np.random.SeedSequence([root,730,seed]))
   cases.append(dict(case_id=f'basic-{scenario}-{seed}',suite='basic',scenario=scenario,execution_seed=seed,mass_ratio=.2,side_m=.045,initial=initial_state(rng,.045)))
 for si,scenario in enumerate(SCENARIOS):
  for zi,size in enumerate((.03,.045,.06)):
   for mi,mass in enumerate((.1,.3)):
    seed=74000+si*6+zi*2+mi;rng=np.random.default_rng(np.random.SeedSequence([root,740,seed]))
    cases.append(dict(case_id=f'endpoint-{scenario}-{zi}-{mi}',suite='endpoints',scenario=scenario,execution_seed=seed,mass_ratio=mass,side_m=size,initial=initial_state(rng,size,2*np.pi*(si*6+zi*2+mi)/18)))
 for si,scenario in enumerate(SCENARIOS):
  for zi,size in enumerate((.03,.045,.06)):
   for j in range(12 if si==zi==0 else 11):
    seed=75000+si*1000+zi*100+j
    mass=np.random.default_rng(np.random.SeedSequence([root,750,seed])).uniform(.1,.3)
    rng=np.random.default_rng(np.random.SeedSequence([root,751,seed]))
    cases.append(dict(case_id=f'heldout-{scenario}-{zi}-{j}',suite='heldout',scenario=scenario,execution_seed=seed,mass_ratio=float(mass),side_m=size,initial=initial_state(rng,size)))
 freeze._formal=cases
 return copy.deepcopy(cases)
