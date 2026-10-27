"""Prospective keyed routing domain with explicit, unopened held-out boundary."""
import copy
import json
from pathlib import Path
import numpy as np
from .dispersed_layout import BOUNDS, layout_checks, coordinate_hash
from .elevated_layout import transform_coordinates
from .scenario import Scenario
from .analytical import content_hash
from .util import json_write

ROOT_SEED = 10262502
SPLITS = {'train':10, 'validation':20, 'held-out':30}
PROFILE = 'routing-elevated3d-v1'
KEY_CONTRACT = dict(root=ROOT_SEED, split_codes=SPLITS,
    fields={'geometry':1,'integer_priorities':2,'cost_multipliers':3,'event_data':4,'training_sampling':5},
    key=['root','split_code','layout_index','field_code','attempt'],
    geometry_attempts=50000, nongeometry_attempt=0,
    training_sampling_key=['root',10,'training_seed',5,'episode_index'], training_layouts=10000)


def keyed_rng(split,index,field,attempt=0):
    if split not in SPLITS or any(type(v) is not int or v < 0 for v in (index,field,attempt)) or field not in range(1,6):
        raise ValueError('Invalid dataset key')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([ROOT_SEED,SPLITS[split],index,field,attempt])))


def training_draw(seed,episode):
    rng = keyed_rng('train',seed,5,episode)
    return int(rng.integers(10000)), str(rng.choice(['static','priority','cost'],p=[.5,.25,.25]))


def generate_case(split,index,kind=None,max_attempts=50000,*,freeze=None):
    if split=='held-out':
        if freeze is None:raise ValueError('Joint freeze required before held-out materialization')
        from .elevated_campaign import JointFreeze
        if not isinstance(freeze,JointFreeze):raise ValueError('Typed joint freeze required')
        freeze.require_current()
    if split not in SPLITS or type(index) is not int or not 0 <= index < (10000 if split=='train' else 300):
        raise ValueError('Invalid split/layout index')
    if type(max_attempts) is not int or not 0 <= max_attempts <= 50000:
        raise ValueError('Invalid geometry budget')
    for attempt in range(max_attempts):
        old = [[0.,0.,1.], *keyed_rng(split,index,1,attempt).uniform(BOUNDS[:,0],BOUNDS[:,1],(10,3)).tolist()]
        if layout_checks(old)['passed']:
            coords = transform_coordinates(old)
            break
    else:
        raise RuntimeError(f'Geometry exhausted: {split}/{index}, {max_attempts} candidates')
    priorities = keyed_rng(split,index,2).integers(1,101,10).tolist()
    costs = keyed_rng(split,index,3).uniform(1,5,(11,11)); np.fill_diagonal(costs,1)
    event_rng = keyed_rng(split,index,4)
    target = str(event_rng.choice(list('ABCDEFGHIJ')))
    if kind is None:
        kind = 'static' if index < 100 else ('priority' if index < 200 else 'cost')
    event = None
    if kind=='priority':
        event = dict(kind='priority',after=1,checkpoint=target,value=100)
    elif kind=='cost':
        # A fixed directed edge, independent of any policy's first action.
        source = str(event_rng.choice([v for v in 'SABCDEFGHIJ' if v!=target]))
        event = dict(kind='cost',after=1,edges=[[source,target]],value=5)
    elif kind!='static':
        raise ValueError('Invalid event stratum')
    s = Scenario(f'{PROFILE}/{ROOT_SEED}/{split}/{index}/{kind}',split,index,coords,priorities,costs.tolist(),
                 [.5]*10,beta=.05,event=event,schema_version=3,checkpoint_ids=list('ABCDEFGHIJ'))
    s.validate_physical()
    # Include a saved independent boundary draw for the training semantics:
    # after first completion choose uniformly from the nine eligible goals;
    # value uniformly 1..100 or 1..5, with no model-seed influence on layout data.
    boundary = dict(eligible_rank=int(event_rng.integers(9)),priority=int(event_rng.integers(1,101)),cost=int(event_rng.integers(1,6)))
    semantic = s.to_dict(); semantic = {k:v for k,v in semantic.items() if k not in ('scenario_id','split','seed')}
    return dict(profile=PROFILE,root=ROOT_SEED,split=split,index=index,kind=kind,selected_attempt=attempt,
                coordinate_sha256=coordinate_hash(coords),content_sha256=content_hash(semantic),
                training_boundary=boundary,scenario=s.to_dict())


class Dataset:
    def __init__(self,directory,validation_count=300):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.seen={}; self.cache={}; self.validation_count=validation_count
        from .util import load_config
        for name in ('normal','emergency','cost_shift'):
            c=load_config(f'configs/demo_scenarios_dispersed/{name}.yaml')['coords']
            self.seen[coordinate_hash(transform_coordinates(c))]=('demo',name)

    def register(self,split,index,case):
        identity=(split,index)
        for field in ('coordinate_sha256','content_sha256'):
            h=case[field]
            if h in self.seen and self.seen[h]!=identity:
                raise ValueError(f'Dataset collision: {identity}, {self.seen[h]}')
            self.seen[h]=identity
        self.cache[identity]=copy.deepcopy(case)

    def get(self,split,index):
        if split=='held-out':
            # Until the physical source/camera prerequisites are frozen, no API
            # of this declaration can materialize held-out. A later campaign
            # reader must validate a fully bound joint-freeze receipt.
            raise ValueError('Joint freeze required before held-out materialization')
        key=(split,index)
        if key not in self.cache:
            path=self.directory/split/f'{index:04d}.json'
            expected=generate_case(split,index,kind='static' if split=='train' else None)
            case=json.loads(path.read_text()) if path.exists() else expected
            if case != expected:
                raise ValueError('Invalid cache identity/content/geometry: '+str(path))
            self.register(split,index,case)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                json_write(path,case)
        return copy.deepcopy(self.cache[key])

    def validation(self):
        return [self.get('validation',i) for i in range(self.validation_count)]
