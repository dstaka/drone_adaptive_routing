"""Verify compact development replay against a closed native B-static mission."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from adaptive_drone_patrol.dual_coordination_artifacts import inventory
from adaptive_drone_patrol.dual_coordination_baselines import StaticBaseline
from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv
from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json
from adaptive_drone_patrol.dual_coordination_raw import read_raw
from adaptive_drone_patrol.util import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--original',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--journal-root',required=True)
    p.add_argument('--native-audit-library')
    args=p.parse_args()
    source=Path(args.source).resolve()
    before=inventory(source)
    metadata=json.loads((source/'environment.json').read_text())
    expected=json.loads((source/'pilot-result.json').read_text())
    actions=json.loads((source/'proposed-actions.json').read_text())
    if metadata['mode']!='development' or expected['method']!='B-static':
        raise ValueError('Closed B-static development mission required')
    journal=RunJournal(args.journal_root)
    attempt=journal.start('native-compact-replay-verification',source.name,category='development replay verification')
    env=None
    try:
        env=DualCoordinationEnv(args.original,args.output,metadata['case'],metadata['design'],replay_stride_ticks=20,
                                native_audit_library=args.native_audit_library)
        baseline=StaticBaseline(env.case['starts'],env.case['goals'])
        last=time.monotonic()
        for item in actions:
            if env.terminated or item['tick']!=env.mission.tick:
                raise ValueError('Action clock or terminal mismatch')
            actual=baseline.select(env.mission,[r['drone_xyz'] for r in env.fixture.sample()['drones']]).action
            if actual!=item['decision']['action']:
                raise ValueError('Recomputed baseline decision mismatch')
            _,reward,_,info=env.step(actual)
            if abs(reward-item['reward'])>1e-10 or info['terminal_reason']!=item['terminal_reason']:
                raise ValueError('Reward or terminal decision mismatch')
            if time.monotonic()-last>=30:
                journal.heartbeat(attempt,completed=env.mission.tick,total=round(expected['team_time']*100),reason='Every native stage audited; matching original proposed decisions')
                last=time.monotonic()
        env.close()
        originals={r['tick']:r for path in sorted(source.glob('replay-*.draw')) for r in read_raw(path)}
        maximum=0.
        checked=0
        for segment in env.replay_segments:
            for row in read_raw(segment['path']):
                original=originals[row['tick']]
                for field in ('qpos','qvel','rpm','ctrl'):
                    error=float(np.max(np.abs(np.asarray(row['actual'][field])-np.asarray(original['actual'][field]))))
                    maximum=max(maximum,error)
                    if error>1e-12:
                        raise ValueError('Native state replay mismatch: '+field)
                checked+=1
        metrics=dict(team_time=env.mission.tick*.01,completed_counts=env.mission.completed_counts,
                     minimum_control_boundary_clearance_m=env.ledger.minimum_clearance_m,
                     floor_minima_m=env.ledger.floor_minima_m,terminal_reason=env.mission.terminal_reason)
        for field,value in metrics.items():
            if isinstance(value,str) or field=='completed_counts':
                if value!=expected[field]:
                    raise ValueError('Metric mismatch: '+field)
            elif not np.allclose(value,expected[field],atol=1e-9,rtol=0):
                raise ValueError('Metric mismatch: '+field)
        if inventory(source)!=before:
            raise ValueError('Original closed artifacts changed during verification')
        result=dict(schema='dual-development-replay-verification-v1',verified=True,
                    production_qualified=False,source_inventory=before,source_profile=metadata.get('replay_stride_ticks',1),
                    replay_stride_ticks=20,decisions_match=True,terminal_match=True,metrics_within_tolerance=True,
                    checked_states=checked,maximum_state_absolute_error=maximum,metrics=metrics,
                    decision_count=len(actions),audit_library_sha256=sha256(args.native_audit_library) if args.native_audit_library else None,
                    retained_replay_bytes=sum(Path(r['path']).stat().st_size for r in env.replay_segments),
                    source_action_sha256=sha256(source/'proposed-actions.json'),online_audit=env.online_auditor.report())
        atomic_json(Path(args.output)/'replay-verification.json',result)
        journal.finish(attempt,status='passed',reason=f'{len(actions)} decisions and {checked} retained states matched; maximum error{maximum}')
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_inventory','online_audit')},indent=2))
    except BaseException as error:
        journal.finish(attempt,status='failed',reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__=='__main__':
    main()
