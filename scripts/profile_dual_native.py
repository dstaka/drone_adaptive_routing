"""Disposable development profile; no training or qualification admission."""
import cProfile
import json
from pathlib import Path
import pstats
from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv
from adaptive_drone_patrol.dual_coordination_schema import encode_action
case=json.loads(Path('../run/development-challenges.json').read_text())[0]['case']
env=DualCoordinationEnv('inputs/cf2x/cf2x.urdf','../run/native-profile-001',case,json.loads(Path('inputs/controller.json').read_text()))
profile=cProfile.Profile()
profile.enable()
env.step(encode_action(10,10,1))
profile.disable()
env.close()
profile.dump_stats('../run/native-profile-001/step.prof')
pstats.Stats(profile).strip_dirs().sort_stats('cumulative').print_stats(35)
