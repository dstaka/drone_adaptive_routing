"""Fresh human review windows must span actual events through next shared stop."""

import pytest


def test_event_review_window_covers_before_event_to_next_shared_settling():
    from adaptive_drone_patrol.payload_solver_capture import review_windows

    events = [dict(type='SHARED_SETTLED', simulation_step=150),
              dict(type='PRIORITY_UPDATED', simulation_step=150),
              dict(type='TARGET_SELECTED', simulation_step=150),
              dict(type='SHARED_SETTLED', simulation_step=499)]
    assert review_windows(events, 1000) == [dict(event='PRIORITY_UPDATED',
        event_step=150, from_step=50, through_step=499, from_s=.5, through_s=4.99)]


def test_event_without_a_following_actual_shared_stop_cannot_enter_video_review():
    from adaptive_drone_patrol.payload_solver_capture import review_windows

    with pytest.raises(ValueError, match='next'):
        review_windows([dict(type='COST_UPDATED', simulation_step=300)], 1000)


def test_saved_success_labels_cannot_authorize_public_capture(tmp_path, monkeypatch):
    import json
    from test_payload_recovery_mission import run
    from adaptive_drone_patrol import payload_solver_capture as capture

    trial = tmp_path / 'fixed' / 'basic-normal-1'
    trial.mkdir(parents=True)
    _, mission, _ = run(trial)
    mission.rename(trial / 'mission')
    (trial / 'independent-check.json').write_text(json.dumps(dict(success=True, physical_acceptance=True)))
    font = tmp_path / 'font.ttf'
    freeze = dict(artifact_sha256={'untrusted': '0' * 64}, camera_assets={str(font): '0' * 64},
                  models=dict(randomized=dict(sha256='a' * 64)))
    monkeypatch.setattr(capture, 'verify_file_snapshot', lambda value: True)
    monkeypatch.setattr(capture, 'NativeRenderer', lambda *a, **kw: pytest.fail(
        'Saved status or a fixed trial reached rendering without fresh paired replay'))
    with pytest.raises(ValueError, match='Fresh paired'):
        capture.capture_case(trial=trial, case=dict(case_id='basic-normal-1', scenario='normal',
            mass_ratio=.2, side_m=.045), freeze=freeze, output=tmp_path / 'video', font=font)
    assert not (tmp_path / 'video').exists()


@pytest.mark.parametrize('trials,accepted', [(253, True), (254, False)])
def test_public_capture_batch_rechecks_the_whole_acceptance_before_output(
        tmp_path, monkeypatch, trials, accepted):
    import json
    from adaptive_drone_patrol import payload_solver_capture as capture
    from adaptive_drone_patrol import payload_solver_trials

    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps({'cases': str(tmp_path / 'not-opened.json')}))
    calls = []
    def recheck(**kwargs):
        calls.append(kwargs)
        return dict(trials=trials, statistics=dict(randomized_acceptance=accepted))
    monkeypatch.setattr(payload_solver_trials, 'recheck_paired_trials', recheck)
    with pytest.raises(ValueError, match='Fresh paired254'):
        capture.capture_all_cases(freeze_path=freeze, trials=tmp_path / 'trials',
            admission=object(), planner=object(), font=tmp_path / 'font', output=tmp_path / 'video')
    assert len(calls) == 1 and not (tmp_path / 'video').exists()


def test_fresh_paired_proof_cannot_authorize_a_fixed_trial_as_randomized(tmp_path, monkeypatch):
    import json
    from adaptive_drone_patrol import payload_solver_capture as capture
    from adaptive_drone_patrol import payload_solver_trials

    case = dict(case_id='basic-normal-1', scenario='normal', mass_ratio=.2, side_m=.045)
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps(dict(cases=[case])))
    frozen = dict(cases=str(cases))
    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps(frozen))
    calls = []
    def recheck(**kwargs):
        calls.append(kwargs)
        return dict(trials=254, statistics=dict(randomized_acceptance=True), results=[])
    monkeypatch.setattr(payload_solver_trials, 'recheck_paired_trials', recheck)
    monkeypatch.setattr(capture, 'NativeRenderer', lambda *a, **kw: pytest.fail('Wrong policy reached rendering'))
    with pytest.raises(ValueError, match='randomized trial/case'):
        capture.capture_case(trial=tmp_path / 'trials/fixed/basic-normal-1', case=case,
            freeze=frozen, freeze_path=freeze, trials=tmp_path / 'trials', admission=object(),
            planner=object(), output=tmp_path / 'video', font=tmp_path / 'font')
    assert len(calls) == 1 and not (tmp_path / 'video').exists()


@pytest.mark.parametrize('mutation_timing', ['after_replay', 'before_snapshot'])
def test_changed_next_video_geometry_cannot_become_a_new_snapshot_baseline(
        tmp_path, monkeypatch, mutation_timing):
    import json
    from test_payload_recovery_mission import run
    from adaptive_drone_patrol import payload_solver_capture as capture
    from adaptive_drone_patrol import payload_solver_trials
    from adaptive_drone_patrol.util import sha256

    trials = tmp_path / 'trials'
    trial = trials / 'randomized/basic-normal-1'
    trial.mkdir(parents=True)
    _, mission, _ = run(trial)
    mission.rename(trial / 'mission')
    case = dict(case_id='basic-normal-1', scenario='normal', mass_ratio=.2, side_m=.045)
    (trial / 'case.json').write_text(json.dumps(case))
    (trial / 'independent-check.json').write_text(json.dumps(dict(success=True, physical_acceptance=True)))
    native = trial / 'flight/runtime/raw.jsonl.gz'
    native.parent.mkdir(parents=True)
    native.write_bytes(b'CPU binding fixture only, not a physical trace')
    model = trial / 'flight/fixture/model.xml'
    model.parent.mkdir()
    model.write_text('<original-accepted-geometry/>')
    properties = trial / 'flight/runtime-properties.json'
    properties.write_text('{}')
    meta = trial / 'mission/metadata.json'
    metadata = json.loads(meta.read_text())
    metadata['method'] = 'randomized'
    meta.write_text(json.dumps(metadata))
    summary = trial / 'mission/mission_summary.json'
    saved = json.loads(summary.read_text())
    saved['metadata_sha256'] = sha256(meta)
    summary.write_text(json.dumps(saved))
    original = payload_solver_trials.artifact_snapshot(trial)
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps(dict(cases=[case])))
    frozen = dict(cases=str(cases), artifact_sha256={'ignored-by-test-double': '0' * 64},
                  models=dict(randomized=dict(sha256='a' * 64)))
    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps(frozen))
    checked = dict(method='randomized', case_id=case['case_id'], success=True, physical_acceptance=True,
        summary_sha256=sha256(summary), artifact_sha256=original,
        native_proof=dict(checkpoint_sha256='a' * 64, native_raw_sha256=sha256(native)))
    def recheck(**kwargs):
        # Simulate a later video's XML changing after the fresh full replay.
        if mutation_timing == 'after_replay':
            model.write_text('<changed-display-geometry/>')
        return dict(trials=254, statistics=dict(randomized_acceptance=True), results=[checked])
    original_snapshot = payload_solver_trials.artifact_snapshot
    def snapshot(root):
        if mutation_timing == 'before_snapshot':
            model.write_text('<changed-display-geometry/>')
        return original_snapshot(root)
    monkeypatch.setattr(payload_solver_trials, 'artifact_snapshot', snapshot)
    monkeypatch.setattr(payload_solver_trials, 'recheck_paired_trials', recheck)
    monkeypatch.setattr(capture, 'verify_file_snapshot', lambda value: True)
    def render(**kwargs):
        pytest.fail('Changed geometry reached rendering after fresh acceptance replay')
    monkeypatch.setattr(capture, '_render_case', render)
    with pytest.raises(ValueError, match='[Aa]ccepted trial artifact'):
        capture.capture_case(trial=trial, case=case, freeze=frozen, freeze_path=freeze,
            trials=trials, admission=object(), planner=object(), output=tmp_path / 'video', font=tmp_path / 'font')
