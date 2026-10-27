import copy
import numpy as np
import pytest
from adaptive_drone_patrol.elevated_routing_data import keyed_rng, generate_case, Dataset, training_draw
from adaptive_drone_patrol.elevated_routing_experiment import select_model
from adaptive_drone_patrol.objective import RoutingState


def test_keyed_geometry_field_independence_and_no_clipping(tmp_path):
    a = generate_case('validation', 0)
    assert a['scenario']['coords'] == generate_case('validation', 0)['scenario']['coords']
    b = generate_case('validation', 0, kind='priority')
    assert a['coordinate_sha256'] == b['coordinate_sha256']
    assert a['scenario']['priorities'] == b['scenario']['priorities']
    assert a['scenario']['multipliers'] == b['scenario']['multipliers']
    assert generate_case('train', 0)['coordinate_sha256'] != a['coordinate_sha256']
    from adaptive_drone_patrol.scenario import Scenario
    s = Scenario.from_dict(a['scenario'])
    obs = RoutingState(s).observation()
    assert obs[26] > 1  # A's actual elevated z/2, not clipped to old domain.
    assert max(np.array(s.coords)[1:,2]) > 4.7
    assert min(np.array(s.coords)[1:,2]) < 2.3
    assert all(1 <= p <= 100 and type(p) is int for p in s.priorities)
    assert a['selected_attempt'] < 50000
    assert keyed_rng('validation',0,2).integers(1,101,10).tolist() == s.priorities


def test_heldout_blocked_and_collisions_fail_without_replacement(tmp_path):
    dataset = Dataset(tmp_path, validation_count=2)
    with pytest.raises(ValueError, match='freeze'): dataset.get('held-out',0)
    v = dataset.get('validation',0)
    # The same geometry under a different identity invalidates a cohort.
    with pytest.raises(ValueError, match='collision'): dataset.register('train',0,v)
    t = dataset.get('train',0)
    assert dataset.get('train',0) == t
    assert training_draw(42,0) == training_draw(42,0)
    assert training_draw(43,0) != training_draw(42,0)


def score(cost=100., ratio=1.05, greedy=110., fixed=130., complete=True):
    return dict(mean_cost=cost, static_mean_ratio=ratio, static_worst_ratio=1.09,
                static_mean_cost=cost, greedy_static_mean=greedy, fixed_static_mean=fixed,
                complete=complete, strata={k:dict(complete=True, mean_cost=cost) for k in ('static','priority','cost')})


def test_selection_exact_one_percent_and_stratum_guard():
    b = score()
    f = score(99.)
    assert select_model({'B':b,'F42':f})['selected'] == 'F42'
    assert select_model({'B':b,'F42':score(99.00001)})['selected'] == 'B'
    bad = score(98.); bad['strata']['cost']['mean_cost'] = 105.00001
    assert select_model({'B':b,'F42':bad})['selected'] == 'B'
    bad['strata']['cost']['mean_cost'] = 105.
    assert select_model({'B':b,'F42':bad})['selected'] == 'F42'


def test_incomplete_baseline_still_guards_complete_strata_and_ties():
    b = score(complete=False); b['strata']['priority']['complete'] = False
    f = score(98.); f['strata']['cost']['mean_cost'] = 106.
    assert select_model({'B':b,'F42':f})['selected'] is None
    f['strata']['cost']['mean_cost'] = 104.
    assert select_model({'B':b,'F44':copy.deepcopy(f),'F43':copy.deepcopy(f),'F42':f})['selected'] == 'F42'
    assert select_model({'B':score(ratio=1.10001),'F42':score(ratio=1.2)})['selected'] is None


def test_event_is_fixed_exogenous_not_policy_dependent():
    case = generate_case('validation',101)
    e = case['scenario']['event']
    assert e['value'] == 100
    assert not e.get('sample_at_boundary',False)
    assert case == generate_case('validation',101)
    with pytest.raises(RuntimeError, match='exhaust'): generate_case('train',0,max_attempts=0)


def test_weight_only_fresh_optimizer_and_full_rollout_accounting(tmp_path):
    import torch
    from adaptive_drone_patrol.elevated_routing_experiment import fresh_candidate, train_stages
    from adaptive_drone_patrol.routing_agent import RoutingAgent
    baseline='outputs/suspended_mujoco_final_20261025_r3a/inputs/routing/seed_44.zip'
    data=Dataset(tmp_path/'data', validation_count=2)
    data.validation()
    original=RoutingAgent(baseline)
    model,env=fresh_candidate(baseline,42,data,n_steps=16,batch_size=8,n_epochs=1)
    assert model.num_timesteps == 0
    assert len(model.policy.optimizer.state) == 0
    assert all(torch.equal(v,original.model.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
    assert model.routing_schema == original.schema
    receipt=train_stages(model,env,requested=17)
    assert receipt['static_actual'] == 32
    assert receipt['mixture_actual'] == 32
    assert receipt['total_actual'] == 64
    assert len(env.samples) >= 6
    assert model.num_timesteps == 64
    assert len(model.policy.optimizer.state)>0


def test_cached_case_identity_and_content_are_verified(tmp_path):
    import json
    path=tmp_path/'train/0000.json'; path.parent.mkdir()
    original=generate_case('train',0)
    for damage in ('identity','hash','coordinates','field'):
        bad=copy.deepcopy(original)
        if damage=='identity':bad['split']='validation'
        if damage=='hash':bad['coordinate_sha256']='f'*64
        if damage=='coordinates':bad['scenario']['coords'][1][2]=99.
        if damage=='field':bad['scenario']['priorities'][0]=101
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='cache'): Dataset(tmp_path,validation_count=1).get('train',0)


def test_unavailable_metrics_are_rejected_without_nan_or_guard_loss():
    import json
    b=score(complete=False)
    b.update(mean_cost=None,static_mean_cost=None,static_mean_ratio=None,static_worst_ratio=None)
    b['strata']['static']=dict(complete=False,mean_cost=None)
    f=score(99.); f['strata']['cost']['mean_cost']=106.
    result=select_model({'B':b,'F42':f})
    assert result['selected'] is None
    assert 'cost: exceeds baseline 5% guard' in result['rejections']['F42']
    json.dumps(result,allow_nan=False)


def test_low_level_heldout_generation_also_requires_joint_freeze():
    # Invalid index intentionally prevents accidental real held-out exposure
    # while testing the not-yet-implemented guard.
    with pytest.raises(ValueError,match='freeze'):
        generate_case('held-out',-1)


def test_cost_event_to_completed_destination_is_reported_inapplicable():
    from adaptive_drone_patrol.scenario import generate_scenario,INTEGER_PROFILE
    from adaptive_drone_patrol.benchmark import evaluate_scenario
    from adaptive_drone_patrol.elevated_routing_experiment import counterfactual_response
    s=generate_scenario(10000,'validation',num_checkpoints=10,priority_profile=INTEGER_PROFILE)
    s.event=dict(kind='cost',after=1,edges=[['F','A']],value=5)
    fixed=lambda st:int(np.flatnonzero(st.mask())[0])
    r=evaluate_scenario(s,fixed)
    c=counterfactual_response(s,r)
    assert not c['applicable'] and c['reason']=='no eligible remaining directed edge'
    assert r['events'][0]['edges']==[['F','A']]  # Exogenous update remains preserved.
    s.event['edges']=[['A','B']]
    c=counterfactual_response(s,evaluate_scenario(s,fixed))
    assert c['applicable'] and c['response']=='no_change'
