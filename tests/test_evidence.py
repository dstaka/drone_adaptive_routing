import json
import pytest
from adaptive_drone_patrol.evidence import validate_run, map_presentation_time, verify_manifest
from adaptive_drone_patrol.util import sha256


def test_timeline_2x_and_cuts_map_capture_times():
    segments = [{'presentation_start_s': 0, 'presentation_end_s': 5, 'source_start_s': 2,
                 'source_end_s': 12, 'playback_rate': 2, 'run_id': 'r'},
                {'presentation_start_s': 5, 'presentation_end_s': 10, 'source_start_s': 20,
                 'source_end_s': 25, 'playback_rate': 1, 'run_id': 'r'}]
    assert map_presentation_time(segments, 3) == ('r', 8)
    assert map_presentation_time(segments, 5) == ('r', 20)
    with pytest.raises(ValueError):
        map_presentation_time(segments, 11)


def test_manifest_missing_mismatched_and_path_escape_fail(tmp_path):
    file = tmp_path / 'actual.txt'
    file.write_text('real evidence')
    manifest = {'artifacts': [{'path': 'actual.txt', 'sha256': sha256(file), 'bytes': file.stat().st_size}]}
    assert verify_manifest(manifest, tmp_path) == 1
    file.write_text('changed')
    with pytest.raises(ValueError, match='integrity'):
        verify_manifest(manifest, tmp_path)
    manifest['artifacts'][0]['path'] = '../outside'
    with pytest.raises(ValueError, match='outside'):
        verify_manifest(manifest, tmp_path)


def test_false_visit_log_cannot_validate(tmp_path):
    (tmp_path / 'mission_summary.json').write_text(json.dumps({'completed_order': ['A'], 'status': 'SUCCEEDED'}))
    (tmp_path / 'events.jsonl').write_text(json.dumps({'event_seq': 1, 'type': 'MISSION_FINISHED',
                                                     'sim_time_s': 1}) + '\n')
    with pytest.raises(ValueError, match='service'):
        validate_run(tmp_path)
