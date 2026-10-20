"""Integer-profile boundary, objective and compatibility regressions."""

import copy
import numpy as np
import pytest
from adaptive_drone_patrol.scenario import Scenario, generate_scenario
from adaptive_drone_patrol.objective import RoutingState, schema_for
from adaptive_drone_patrol.baselines import exact_remaining, remaining_cost


def integer_scenario(n=10, seed=20001, split='static', kind=None):
    return generate_scenario(seed, split, kind, n, priority_profile='integer-100-v1')


def integer_data():
    data = generate_scenario(20001, 'static', num_checkpoints=3).to_dict()
    data.update(schema_version=3, priorities=[1, 7, 100], beta=.05)
    return data


def test_integer_tokens_survive_snapshots_and_fixed_normalization():
    s = Scenario.from_dict(integer_data())
    assert s.priorities == [1, 7, 100] and all(type(p) is int for p in s.priorities)
    state = RoutingState(s)
    start, end = schema_for(s)['fields']['priorities']
    np.testing.assert_allclose(state.observation()[start:end], [.01, .07, 1.])
    s.priorities[-1] = 50
    np.testing.assert_allclose(RoutingState(s).observation()[start:end], [.01, .07, .5])
    assert schema_for(s)['id'] != schema_for(generate_scenario(20001, 'static'))['id']


@pytest.mark.parametrize('bad', [0, 101, 1.5, 7.0, True, '7', None, float('nan'), float('inf')])
def test_integer_profile_rejects_invalid_tokens_before_coercion(bad):
    data = integer_data()
    data['priorities'][1] = bad
    with pytest.raises(ValueError, match='integer|Integer'):
        Scenario.from_dict(data)


def test_integer_priority_events_and_cost_events_have_separate_domains():
    data = integer_data()
    data['event'] = {'kind': 'priority', 'after': 1, 'checkpoint': 'C', 'value': 100}
    state = RoutingState(Scenario.from_dict(data))
    state.complete(0, state.departure(0))
    record = state.apply_event()
    assert record['old'] == 100 and type(record['value']) is int
    assert state.scenario.priorities == [1, 7, 100]
    for bad in [100., True, 101, '100']:
        data['event']['value'] = bad
        with pytest.raises(ValueError):
            Scenario.from_dict(data)
    data['event'] = {'kind': 'cost', 'after': 1, 'edges': [['A', 'B']], 'value': 100}
    with pytest.raises(ValueError):
        Scenario.from_dict(data)


def test_integer_draws_are_reproducible_independent_and_with_replacement():
    s = integer_scenario()
    again = integer_scenario()
    assert s.to_dict() == again.to_dict()
    old = generate_scenario(20001, 'static', num_checkpoints=10)
    assert s.coords == old.coords and s.multipliers == old.multipliers
    # Changing geometry count does not consume the dedicated priority stream.
    assert integer_scenario(12).priorities[:10] == s.priorities
    draws = [integer_scenario(seed=20000+i).priorities for i in range(100)]
    assert any(len(set(p)) < len(p) for p in draws)
    assert {p for row in draws for p in row} == set(range(1, 101))


def test_integer_objective_uses_raw_and_weighted_components_without_prefix():
    data = integer_data()
    data.update(coords=[[0, 0, 1], [.5, 0, 1], [2, 0, 1], [3, 0, 1]],
                priorities=[20, 100, 40], multipliers=np.ones((4, 4)).tolist())
    state = RoutingState(Scenario.from_dict(data))
    leg = state.departure(0)
    assert leg['completion_penalty'] == 20 and leg['cost'] == pytest.approx(1.5)
    assert leg['weighted_completion_penalty'] == 1
    state.complete(0, leg)
    result = remaining_cost(state, [1, 2])
    # Absolute completion times are 3 and 4.5; prefix cost 1.5 is excluded.
    assert result['completion_penalty'] == 480
    assert result['cost'] == pytest.approx(26.5)
    exact = exact_remaining(state)
    assert exact['cost'] <= result['cost']


def test_legacy_fractional_values_and_profile_coefficients_remain_explicit():
    old = generate_scenario(20001, 'static').to_dict()
    old['priorities'] = [2.12, 2.56, 2.27]
    assert Scenario.from_dict(old).priorities == old['priorities']
    for field, value in [('beta', 1), ('alpha', .5), ('return_to_base', True),
                         ('nominal_speed_mps', 2), ('service_duration_s', [.4]*3)]:
        data = copy.deepcopy(integer_data())
        data[field] = value
        with pytest.raises(ValueError):
            Scenario.from_dict(data)


def test_new_training_checkpoint_rejects_old_semantics_and_tampered_sidecar(tmp_path):
    import json
    from adaptive_drone_patrol.routing_agent import train_one, RoutingAgent
    from adaptive_drone_patrol.planner_ipc import PlannerClient
    import sys
    path = train_one(42, tmp_path / 'integer', static_steps=16, event_steps=16, n_steps=16,
                     batch_size=8, n_epochs=1, policy_arch='candidate',
                     priority_profile='integer-100-v1', warmstart_cases=6, warmstart_epochs=1)
    agent = RoutingAgent(path)
    state = RoutingState(integer_scenario())
    with PlannerClient([sys.executable, '-m', 'adaptive_drone_patrol.planner_ipc', '--model', str(path),
                        '--hash', agent.hash], agent.hash, schema=agent.schema, timeout=30) as client:
        assert client.predict(state) == agent.predict(state)
    with pytest.raises(ValueError, match='retrain'):
        agent.predict(RoutingState(generate_scenario(20001, 'static', num_checkpoints=10)))
    path.with_suffix('.schema.json').write_text(json.dumps(schema_for(
        generate_scenario(0, 'training', num_checkpoints=10))))
    with pytest.raises(ValueError, match='embedded|contract'):
        RoutingAgent(path)


def test_integer_training_rejects_shape_compatible_legacy_teacher(tmp_path):
    from adaptive_drone_patrol.routing_teacher import prepare_teacher
    from adaptive_drone_patrol.routing_agent import train_one
    teacher = prepare_teacher(tmp_path / 'old', 10, 4)
    with pytest.raises(ValueError, match='teacher|Teacher|contract'):
        train_one(42, tmp_path / 'bad', static_steps=16, event_steps=16, n_steps=16,
                  batch_size=8, n_epochs=1, policy_arch='candidate', priority_profile='integer-100-v1',
                  warmstart_cases=4, warmstart_epochs=1, teacher_directory=teacher)


def test_demo_keeps_geometry_and_uses_dedicated_uniform_draws():
    from adaptive_drone_patrol.analytical import demo_fixture
    s = demo_fixture(40004, priority_profile='integer-100-v1')
    assert s.coords == demo_fixture(40004).coords
    assert s.priorities == integer_scenario(seed=40004, split='demo').priorities
    assert all(type(p) is int for p in s.priorities)


def test_integer_palette_and_labels_are_fixed_across_missions():
    from adaptive_drone_patrol.scene import priority_color, priority_text, scene_manifest
    s = integer_scenario()
    assert priority_text(100, 3) == '100'
    assert priority_text(7, 3) == '7'
    assert priority_text(2.56, 2) == '2.56'
    assert priority_color(1, priority_max=100) == '#fee5d9'
    assert priority_color(100, priority_max=100) == '#a50f15'
    before = scene_manifest(s)
    s.priorities = [100]*10
    after = scene_manifest(s)
    assert before['value_colors'] == after['value_colors']
    assert len(before['value_colors']) == 100 and before['priority_range'] == [1, 100]
    with pytest.raises(ValueError):
        priority_color(7., priority_max=100)


@pytest.mark.parametrize('low, qualifies', [(90, True), (91, False)])
def test_feature_requires_ten_point_gap_and_id_sorted_maximum_tie(low, qualifies):
    from adaptive_drone_patrol.analytical import explain_decision, verify_explanation
    s = Scenario('gap', 'validation', 10001,
                 [[0, 0, 1], [.1, 0, 1], [3, 0, 1], [2, 0, 1]], [low, 100, 100],
                 np.ones((4,4)).tolist(), [.5]*3, schema_version=3,
                 checkpoint_ids=['L', 'Z', 'H'], beta=.05)
    def chooser(st):
        return exact_remaining(st)['order'][0]
    value = explain_decision(RoutingState(s), chooser, 'hash', 'run', 'decision')
    assert value['nearer_lower_priority_gate'] == qualifies
    assert value['selected_H'] == ('H' if qualifies else None)
    assert value['maximum_priority_ids'] == ['H', 'Z']
    assert verify_explanation(value, 'hash', 'run', 'decision')


def test_integer_flight_pixels_and_independent_event_caption():
    from PIL import Image
    from matplotlib import font_manager
    from adaptive_drone_patrol.scene import scene_manifest
    from adaptive_drone_patrol.analytical import demo_fixture
    from adaptive_drone_patrol.video_render import Renderer
    from adaptive_drone_patrol.video_check import verify_text_region, event_lines
    s = demo_fixture(40004, priority_profile='integer-100-v1')
    record = {'scenario': s.to_dict(), 'events': [], 'scene': scene_manifest(s),
              'telemetry': [], 'initial': [0,0,1], 'schedule': {'dt': .01}}
    renderer = Renderer(font_manager.findfont('DejaVu Sans'))
    image, _ = renderer.flight(Image.new('RGB', (1280,720)), record,
                               {'simulation_step': 0, 'sim_time_s': 0})
    assert verify_text_region(image, f'A  優先度 {s.priorities[0]}', (1377,726), renderer.fonts[23]) == 0
    event = {'type':'PRIORITY_UPDATED', 'sim_time_s':4, 'update':{'old':7,'value':100},
             'comparison': {'no_event_order':[0], 'post_event_order':[0],
                            'retained_order_post_event':{'cost':11},
                            'replanned_order_post_event':{'cost':10}, 'improvement':1}}
    line = event_lines(event, ['C'], schema_version=3)[0]
    assert '優先度 7 → 100' in line and '7.000' not in line


def test_integer_comparison_card_prints_raw_penalty_and_point_zero_five():
    from PIL import Image, ImageDraw
    from matplotlib import font_manager
    from adaptive_drone_patrol.analytical import explain_decision
    from adaptive_drone_patrol.video_render import Renderer
    from adaptive_drone_patrol.video_check import verify_text_region, verify_card_pixels
    s = Scenario('card', 'validation', 10001,
                 [[0,0,1],[.1,0,1],[2,0,1],[3,0,1]], [60,100,40],
                 np.ones((4,4)).tolist(), [.5]*3, beta=.05, schema_version=3,
                 checkpoint_ids=['L','H','X'])
    def choose(st):
        return exact_remaining(st)['order'][0]
    value = explain_decision(RoutingState(s), choose, 'hash', 'r', 'd')
    value['physical_reference'] = {'simulation_step':0}
    candidates = {c['id']: c for c in value['candidate_priorities_distances']}
    rows = []
    for name in ['L','H']:
        candidate = dict(candidates[name], pose_distance_m=candidates[name]['distance_m'])
        result = next(q for q in value['forced_first'] if q['action'] == candidate['action'])
        rows.append({'label':name,'candidate':candidate,'result':result})
    rows.append({'label':'AI','result':value['policy_rollout']})
    spec = {'kind':'tradeoff','feature':value,'rows':rows,'run_id':'r','decision_id':'d','timestamp_s':0}
    font_path = font_manager.findfont('DejaVu Sans')
    renderer = Renderer(font_path)
    image = Image.new('RGB',(1920,1080),'#102333')
    renderer.tradeoff(ImageDraw.Draw(image),spec)
    result = rows[0]['result']
    line = (f"総コスト {result['cost']:.3f} ＝ 移動 {result['travel_cost']:.3f} ＋ 0.05 × 重要度時刻 "
            f"{result['completion_penalty']:.3f}（{.05 * result['completion_penalty']:.3f}）")
    assert verify_text_region(image,line,(65,344),renderer.fonts[28]) == 0
    assert verify_card_pixels(image,spec,font_path) == 0


@pytest.mark.parametrize('field', ['weighted_travel_cost','weighted_completion_penalty'])
def test_weighted_components_are_verified_not_trusted(field):
    from adaptive_drone_patrol.analytical import explain_decision, verify_explanation
    state = RoutingState(Scenario.from_dict(integer_data()))
    def choose(st):
        return exact_remaining(st)['order'][0]
    value = explain_decision(state, choose, 'hash', 'r', 'd')
    value['policy_rollout'][field] = 123456.
    with pytest.raises(ValueError, match='component'):
        verify_explanation(value,'hash','r','d')


@pytest.mark.parametrize('n', range(3,8))
@pytest.mark.parametrize('with_prefix', [False,True])
def test_integer_exact_and_forced_first_match_independent_enumeration(n, with_prefix):
    from itertools import permutations
    import math
    state = RoutingState(integer_scenario(n))
    if with_prefix:
        state.complete(0,state.departure(0))
    s = state.scenario
    costs = []
    for order in permutations(np.flatnonzero(state.mask()).tolist()):
        current, elapsed = state.current, state.elapsed
        travel = penalty = 0.
        for action in order:
            distance = math.dist(s.coords[current],s.coords[action+1])
            elapsed += distance + .5
            travel += distance*s.multipliers[current][action+1]
            penalty += s.priorities[action]*elapsed
            current = action+1
        costs.append((travel+.05*penalty,order))
    assert exact_remaining(state)['cost'] == pytest.approx(min(costs)[0])
    for first in np.flatnonzero(state.mask()).tolist():
        actual = exact_remaining(state,first=first)
        assert actual['cost'] == pytest.approx(min(cost for cost,order in costs if order[0]==first))
        assert actual['weighted_completion_penalty'] == pytest.approx(.05*actual['completion_penalty'])
