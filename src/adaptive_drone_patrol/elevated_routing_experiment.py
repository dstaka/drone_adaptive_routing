"""Finite routing comparison; the prospective validation selection rule."""
import math


def select_model(scores):
    def absolute(s):
        values=[s[k] for k in ('mean_cost','static_mean_ratio','static_worst_ratio','static_mean_cost','greedy_static_mean','fixed_static_mean')]
        return (all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values) and s['complete']
                and s['static_mean_ratio']<=1.10 and s['static_mean_cost']<=s['greedy_static_mean']
                and s['static_mean_cost']<s['fixed_static_mean'])
    b=scores['B']; eligible={}; reasons={}
    for name,s in scores.items():
        ok=absolute(s); rejected=[]
        if not ok: rejected.append('absolute completion/validity/static-quality gate')
        if name!='B':
            for kind,base in b['strata'].items():
                if base['complete'] and (not s['strata'][kind]['complete'] or s['strata'][kind]['mean_cost']>base['mean_cost']*1.05):
                    ok=False; rejected.append(f'{kind}: exceeds baseline 5% guard')
        eligible[name]=ok; reasons[name]=rejected
    candidates=[k for k in scores if k!='B' and eligible[k]]
    best=min(candidates,key=lambda n:(scores[n]['mean_cost'],scores[n]['static_worst_ratio'],int(n[1:]))) if candidates else None
    chosen='B' if eligible['B'] else best
    if best and (not eligible['B'] or scores[best]['mean_cost']<=b['mean_cost']*.99): chosen=best
    return dict(selected=chosen,eligible=eligible,rejections=reasons,scores=scores,
                status='SELECTED' if chosen else 'ROUTING_REQUIREMENT_UNMET',heldout_opened=False)


# All effective optimizer/PPO settings are persisted before campaign data draws.
BASELINE_SHA256 = 'a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef'

PPO_SETTINGS = dict(gamma=1.,gae_lambda=.95,learning_rate=.0003,n_steps=512,batch_size=128,
                    n_epochs=5,clip_range=.2,clip_range_vf=None,ent_coef=0.,vf_coef=.5,
                    max_grad_norm=.5,normalize_advantage=True,target_kl=None,device='cpu')


def fresh_candidate(baseline,seed,dataset,*,n_steps=512,batch_size=128,n_epochs=5):
    import copy
    import numpy as np
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from .routing_agent import RoutingAgent
    from .routing_env import RoutingEnv
    from .elevated_routing_data import training_draw
    from .scenario import Scenario
    from .objective import RoutingState
    torch.set_num_threads(1)
    base=RoutingAgent(baseline,BASELINE_SHA256)

    class TrainingEnv(RoutingEnv):
        def __init__(self):
            super().__init__(num_checkpoints=10,priority_profile='integer-100-v1')
            self.episode_index=0; self.samples=[]; self.boundary=None
        def reset(self,*,seed=None,options=None):
            gym.Env.reset(self,seed=seed)
            index,kind=training_draw(training_seed,self.episode_index)
            if not self.event_mixture: kind='static'
            case=dataset.get('train',index)
            s=Scenario.from_dict(case['scenario'])
            s.scenario_id += f'/episode-{self.episode_index}/{kind}'
            if kind=='priority': s.event=dict(kind=kind,after=1,checkpoint='C',value=100)
            if kind=='cost': s.event=dict(kind=kind,after=1,edges=[['F','A']],value=5)
            self.state=RoutingState(s); self.steps=0; self.finished=False
            self.boundary=case['training_boundary']
            self.samples.append(dict(episode=self.episode_index,index=index,kind=kind,
                                     coordinate_sha256=case['coordinate_sha256']))
            self.episode_index+=1
            return self.state.observation(),{}
        def step(self,action):
            # Same uniform eligible-target/value semantics as legacy training,
            # driven by the independent saved event field for this layout.
            event=self.state.scenario.event
            if not self.state.visited.any() and event and action in range(10):
                eligible=[i for i in range(10) if i!=action]
                target=self.state.scenario.checkpoint_ids[eligible[self.boundary['eligible_rank']]]
                if event['kind']=='priority':
                    event.update(checkpoint=target,value=self.boundary['priority'])
                else:
                    source=self.state.scenario.checkpoint_ids[action]
                    event.update(edges=[[source,target]],value=self.boundary['cost'])
            obs,reward,terminated,truncated,info=super().step(action)
            return obs,reward*.01,terminated,truncated,info
    training_seed=seed
    env=TrainingEnv()
    settings={**PPO_SETTINGS,'n_steps':n_steps,'batch_size':batch_size,'n_epochs':n_epochs}
    model=MaskablePPO(type(base.model.policy),env,seed=seed,
                      policy_kwargs=copy.deepcopy(base.model.policy_kwargs),**settings)
    model.policy.load_state_dict(base.model.policy.state_dict(),strict=True)
    model.routing_schema=copy.deepcopy(base.schema)
    if model.policy.optimizer.state or model.num_timesteps:
        raise ValueError('Candidate must start with fresh optimizer/counters')
    return model,env


def train_stages(model,env,requested=50000,callback=None):
    import math
    rounded=math.ceil(requested/model.n_steps)*model.n_steps
    model.elevated_stage='static'
    model.learn(requested,reset_num_timesteps=True,callback=callback)
    first=model.num_timesteps
    if first!=rounded: raise ValueError('Static rollout accounting mismatch')
    model.elevated_stage='event-mixture'
    env.event_mixture=True
    model.set_env(env)  # force a fresh stage observation, keep episode sampling index
    model.learn(requested,reset_num_timesteps=False,callback=callback)
    second=model.num_timesteps-first
    if second!=rounded: raise ValueError('Event-mixture rollout accounting mismatch')
    return dict(static_requested=requested,mixture_requested=requested,static_actual=first,
                mixture_actual=second,total_actual=model.num_timesteps,terminal_only=True)


def evaluate_models(models,cases,directory,*,input_manifest=None,selected_model=None):
    """All planners share immutable saved inputs; exact dynamic suffixes are online."""
    import numpy as np
    from pathlib import Path
    from .routing_agent import RoutingAgent
    from .scenario import Scenario
    from .objective import RoutingState
    from .benchmark import evaluate_scenario
    from .analytical import rollout, snapshot
    from .baselines import choose_baseline, exact_remaining, remaining_cost
    from .util import json_write, sha256
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    agents={name:RoutingAgent(item['path'],item['sha256']) for name,item in models.items()}
    details=[]; refs={}; model_scores={}
    for i,case in enumerate(cases):
        s=Scenario.from_dict(case['scenario'])
        planners={name:a.predict for name,a in agents.items()}
        planners.update(greedy=lambda st:choose_baseline(st,'greedy'),fixed=lambda st:choose_baseline(st,'fixed'))
        reference=exact_remaining(RoutingState(s)) if case['kind']=='static' else None
        if reference: refs[s.scenario_id]=reference
        results={}
        for name,chooser in planners.items():
            try:
                result=evaluate_scenario(s,chooser)
                state=RoutingState(s); a=int(chooser(state)); state.complete(a,state.departure(a))
                boundaries=[]
                if s.event:
                    for phase in ('pre','post'):
                        if phase=='post': state.apply_event()
                        frozen=snapshot(state)
                        suffix=remaining_cost(state,rollout(state,chooser))
                        optimum=exact_remaining(state)
                        boundaries.append(dict(phase=phase,snapshot=frozen,policy=suffix,exact=optimum,gap=suffix['cost']-optimum['cost']))
                result.update(boundary_exact=boundaries,completed_destinations=len(result['order']),
                              static_exact_ratio=result['analytical_cost']/reference['cost'] if reference else None,
                              static_exact_gap=result['analytical_cost']-reference['cost'] if reference else None)
                result['harmful_responses']=sum(c['applicable'] and c['improvement'] < -1e-6 for c in result['boundary_comparisons'])
                result['no_change_responses']=sum(c['applicable'] and abs(c['improvement'])<=1e-6 for c in result['boundary_comparisons'])
                response=counterfactual_response(s,result);result['route_event_response']=response
                if s.event:
                    result['event_opportunities']=int(response['applicable'] is True)
                    result['skipped_events']=int(response['applicable'] is False)
                    for label in ('beneficial','harmful','no_change'):
                        result[label+'_responses']=int(response['response']==label)
            except (ValueError,RuntimeError) as exc:
                result=dict(error=str(exc),completion_ratio=0.,completed_destinations=0,invalid_actions=1,analytical_cost=None)
            results[name]=result
        details.append(dict(case_id=s.scenario_id,kind=case['kind'],results=results))
        json_write(directory/f'case-{i:03d}.json',details[-1])
        if i%25==0: print(f'evaluation {i+1}/{len(cases)}',flush=True)
    json_write(directory/'exact-static.json',refs)
    for name in models:
        strata={}
        for kind in ('static','priority','cost'):
            rows=[r['results'][name] for r in details if r['kind']==kind]
            valid=bool(rows) and all(r['completion_ratio']==1. and r['invalid_actions']==0 and r['completed_destinations']==10 for r in rows)
            vals=[r['analytical_cost'] for r in rows if r['analytical_cost'] is not None]
            strata[kind]=dict(cases=len(rows),complete=valid,mean_cost=float(np.mean(vals)) if valid else None,
                              p95_cost=float(np.percentile(vals,95)) if valid else None,worst_cost=max(vals) if valid else None)
        static=[r['results'][name] for r in details if r['kind']=='static']
        ratios=[r['static_exact_ratio'] for r in static if r.get('static_exact_ratio') is not None]
        complete=all(x['complete'] for x in strata.values()) and len(details)==300 and all(x['cases']==100 for x in strata.values())
        allcost=[r['results'][name]['analytical_cost'] for r in details if r['results'][name]['analytical_cost'] is not None]
        model_scores[name]=dict(complete=complete,mean_cost=float(np.mean(allcost)) if allcost else None,
            static_mean_ratio=float(np.mean(ratios)) if ratios else None,static_worst_ratio=max(ratios) if ratios else None,
            static_mean_cost=strata['static']['mean_cost'],
            greedy_static_mean=float(np.mean([r['results']['greedy']['analytical_cost'] for r in details if r['kind']=='static'])),
            fixed_static_mean=float(np.mean([r['results']['fixed']['analytical_cost'] for r in details if r['kind']=='static'])),strata=strata,
            invalid_actions=sum(r['results'][name]['invalid_actions'] for r in details),
            mean_static_gap=float(np.mean([r['static_exact_gap'] for r in static])) if strata['static']['complete'] else None,
            p95_cost=float(np.percentile(allcost,95)) if allcost else None,worst_cost=max(allcost) if allcost else None)
    if selected_model is None:
        receipt=select_model(model_scores)
    else:
        s=model_scores[selected_model]
        absolute=(s['complete'] and s['static_mean_ratio'] is not None and s['static_mean_ratio']<=1.10
                  and s['static_mean_cost']<=s['greedy_static_mean'] and s['static_mean_cost']<s['fixed_static_mean'])
        receipt=dict(selected=selected_model,scores=model_scores,absolute_heldout_gate=bool(absolute),
                     status='HELD_OUT_PASSED' if absolute else 'ROUTING_HELD_OUT_UNMET',heldout_opened=True,reselection=False)
    receipt.update(models=models,input_manifest_sha256=sha256(input_manifest or directory.parent/'validation-manifest.json'),
                   exact_reference_sha256=sha256(directory/'exact-static.json'))
    json_write(directory/'selection.json',receipt)
    return receipt


def counterfactual_response(scenario,result):
    """Route applicability, separate from the unchanged logical update log."""
    from .objective import RoutingState
    if not scenario.event:return dict(applicable=False,reason='static',response=None)
    if not result.get('order') or not result.get('boundary_comparisons'):
        return dict(applicable=None,reason='incomplete boundary evidence',response=None)
    state=RoutingState(scenario);first=scenario.checkpoint_ids.index(result['order'][0]);state.complete(first,state.departure(first))
    event=scenario.event;ids=scenario.node_indices
    if event['kind']=='priority':
        applies=not bool(state.visited[ids[event['checkpoint']]-1]);reason='eligible priority target' if applies else 'priority target already completed'
    else:
        def eligible(edge):
            source,target=(ids[k] for k in edge)
            return target>0 and not state.visited[target-1] and (source==state.current or source>0 and not state.visited[source-1])
        applies=any(eligible(edge) for edge in event['edges']);reason='eligible remaining directed edge' if applies else 'no eligible remaining directed edge'
    c=result['boundary_comparisons'][0];delta=float(c['improvement'])
    return dict(applicable=bool(applies),reason=reason,response=('beneficial' if delta>1e-6 else 'harmful' if delta< -1e-6 else 'no_change') if applies else None,
        improvement=delta,first_decision_changed=c['no_event_first']!=c['post_event_first'],suffix_changed=c['no_event_order']!=c['post_event_order'],
        definition='Remaining-cost difference repriced at the same post-event state; no_change means absolute delta<=1e-6; logical update/event bytes unchanged')
