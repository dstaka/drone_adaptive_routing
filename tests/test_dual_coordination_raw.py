"""Lossless storage tests; synthetic arrays are not flight evidence."""
import importlib
import math

import pytest


def module():
    try:
        return importlib.import_module('adaptive_drone_patrol.dual_coordination_raw')
    except ModuleNotFoundError:
        pytest.fail('Lossless dual native archive is not implemented')


def frame(tick=0):
    return dict(schema='dual-raw-v1', tick=tick, world_time=tick*.01, case_id='fixture',
                stages=[dict(qpos=[0., -0., .1, math.nextafter(.1, 1.)],
                             ids=[0, -1, 2**63-1], world_time=tick*.01+i*.0001) for i in range(64)],
                substeps=[dict(geom_xmat=[[1., 0., 0.], [0., 1., 0.]], contacts=[])
                          for _ in range(17)],
                contacts=[] if tick % 2 == 0 else [dict(geom1=1, geom2=2, distance=-.001)],
                huge_integer=2**90, mixed=[1, 1., True, None, '日本語'], empty={}, extra=[[], [1, 2]])


def signature(value):
    # Independent comparison includes scalar types and exact float bit pattern.
    if isinstance(value, dict):
        return [(k, signature(v)) for k, v in value.items()]
    if isinstance(value, list):
        return [signature(v) for v in value]
    return type(value).__name__, value.hex() if type(value) is float else value


def test_exact_roundtrip_all_fields_types_stage_counts_and_variable_contacts(tmp_path):
    m = module()
    original = [frame(i) for i in range(7)]
    path = tmp_path/'mission.drawl'
    with m.RawWriter(path, chunk_frames=3) as writer:
        for row in original:
            writer.append(row)
    recovered = list(m.read_raw(path))
    assert signature(recovered) == signature(original)
    assert len(recovered[0]['stages']) == 64 and len(recovered[0]['substeps']) == 17
    assert m.inspect_raw(path)['frame_count'] == 7
    assert m.inspect_raw(path)['chunk_count'] == 3
    with pytest.raises(FileExistsError):
        m.RawWriter(path)


@pytest.mark.parametrize('damage', ['truncate', 'flip', 'trailing', 'missing_chunk', 'missing_footer'])
def test_corruption_fails_before_any_frame_is_yielded(tmp_path, damage):
    m = module()
    path = tmp_path/'mission.drawl'
    with m.RawWriter(path, chunk_frames=1) as writer:
        writer.append(frame(0))
        writer.append(frame(1))
    original = path.read_bytes()
    info = m.inspect_raw(path)
    if damage == 'truncate':
        damaged = original[:-1]
    elif damage == 'flip':
        changed = bytearray(original)
        changed[info['chunks'][1]['offset']+55] ^= 1
        damaged = bytes(changed)
    elif damage == 'trailing':
        damaged = original+b'extra'
    elif damage == 'missing_chunk':
        damaged = original[:info['chunks'][1]['offset']]+original[info['footer_offset']:]
    else:
        damaged = original[:info['footer_offset']]
    path.write_bytes(damaged)
    with pytest.raises(ValueError):
        next(m.read_raw(path))


def test_writer_rejects_duplicate_time_ticks_mixed_schema_and_identity(tmp_path):
    m = module()
    with m.RawWriter(tmp_path/'invalid.drawl') as writer:
        writer.append(frame(0))
        duplicate = frame(0)
        with pytest.raises(ValueError, match='monotone'):
            writer.append(duplicate)
        bad = frame(1)
        bad['world_time'] = 0.
        with pytest.raises(ValueError, match='monotone'):
            writer.append(bad)
        for key, value in [('schema', 'single-drone-raw-v1'), ('case_id', 'different'), ('tick', -1)]:
            bad = frame(1)
            bad[key] = value
            with pytest.raises(ValueError):
                writer.append(bad)
        writer.append(frame(1))
    assert len(list(m.read_raw(tmp_path/'invalid.drawl'))) == 2


def test_no_ticks_are_permitted_consistently_and_failed_context_has_no_footer(tmp_path):
    m = module()
    path = tmp_path/'no-tick.drawl'
    with m.RawWriter(path) as writer:
        row = frame()
        del row['tick']
        writer.append(row)
        with pytest.raises(ValueError, match='(?i)tick'):
            writer.append(frame(1))
    assert 'tick' not in next(m.read_raw(path))
    abandoned = tmp_path/'abandoned.drawl'
    with pytest.raises(RuntimeError):
        with m.RawWriter(abandoned) as writer:
            writer.append(frame())
            raise RuntimeError('simulated worker crash')
    with pytest.raises(ValueError):
        list(m.read_raw(abandoned))


def test_reject_non_json_nonfinite_and_do_not_finalize_invalid_append(tmp_path):
    m = module()
    with m.RawWriter(tmp_path/'finite.drawl') as writer:
        for value in [float('nan'), float('inf'), object(), {1: 'non-string-key'}]:
            row = frame()
            row['invalid'] = value
            with pytest.raises(ValueError):
                writer.append(row)
        writer.append(frame())
    assert len(list(m.read_raw(tmp_path/'finite.drawl'))) == 1


def test_append_snapshots_mutable_record_and_close_is_idempotent(tmp_path):
    m = module()
    path = tmp_path/'mutable.drawl'
    writer = m.RawWriter(path)
    row = frame()
    writer.append(row)
    row['stages'][0]['qpos'][0] = 123.
    writer.close()
    writer.close()
    assert next(m.read_raw(path))['stages'][0]['qpos'][0] == 0.
    with pytest.raises(ValueError, match='closed'):
        writer.append(frame(1))


def test_probe_reset_step_wrapper_uses_nested_time_without_changing_records(tmp_path):
    m = module()
    reset = dict(kind='RESET', actual=frame(0), geometry={'world_time': 0., 'contacts': []})
    del reset['actual']['tick']
    step = dict(kind='STEP', tick=1, actual=frame(1), stages=frame(1)['stages'], substeps=frame(1)['substeps'])
    del step['actual']['tick']
    path = tmp_path/'probe.drawl'
    with m.RawWriter(path) as writer:
        writer.append(reset)
        writer.append(step)
    assert signature(list(m.read_raw(path))) == signature([reset, step])
    with m.RawWriter(tmp_path/'wrong-reset.drawl') as writer:
        writer.append(reset)
        with pytest.raises(ValueError):
            writer.append(reset)
        stale = dict(step)
        stale['actual'] = dict(step['actual'], schema='old-raw-v1')
        with pytest.raises(ValueError):
            writer.append(stale)
