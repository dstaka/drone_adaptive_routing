"""Development diagnostic of strict JSON incompatibility; no coercion or repair."""
import json
from pathlib import Path
from adaptive_drone_patrol import dual_coordination_raw as raw
from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv
from adaptive_drone_patrol.dual_coordination_cases import generate_case
from adaptive_drone_patrol.dual_coordination_schema import encode_action

original_append=raw.RawWriter.append

def inspect(value,path='$'):
    if type(value) is dict:
        for k,v in value.items():
            yield from inspect(v,path+'.'+str(k))
    elif type(value) is list:
        for i,v in enumerate(value):
            yield from inspect(v,path+f'[{i}]')
    elif value is not None and type(value) not in (str,bool,int,float):
        yield dict(path=path,type=str(type(value)),value=str(value)[:150])

def diagnose(self,record):
    try:
        return original_append(self,record)
    except ValueError as e:
        print(json.dumps(dict(error=str(e),cause=repr(e.__cause__),incompatible=list(inspect(record))),indent=2),flush=True)
        raise
raw.RawWriter.append=diagnose
case=generate_case(15,0,'normal')
case['starts']=[[-3.5,-3.5,1.],[3.5,3.5,1.]]
case['initials']=[dict(position=p,quaternion_wxyz=[1.,0.,0.,0.],swing_deg=0.,direction_rad=0.) for p in case['starts']]
env=DualCoordinationEnv('inputs/cf2x/cf2x.urdf','../run/raw-type-diagnostic-001',case,json.loads(Path('inputs/controller.json').read_text()))
print(env.step(encode_action(10,10,1))[3]['terminal_reason'])
env.close()
