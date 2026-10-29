"""Real software encoder checks; native rendering is a separate explicit check."""

import importlib
import json

import numpy as np
import pytest


def module():
    return importlib.import_module("adaptive_drone_patrol.dual_coordination_media_run")


def pixels(count=5):
    for index in range(count):
        native = np.arange(12 * 24 * 3, dtype=np.uint8).reshape(12, 24, 3) + np.uint8(index)
        hud = np.concatenate([native, np.full((4, 24, 3), index * 20, dtype=np.uint8)])
        yield native, hud, dict(index=index, time_s=index / 100, views={})


def test_real_ffv1_roundtrip_preserves_every_native_and_hud_pixel_and_full_1x_timeline(tmp_path):
    m = module()
    result = m.encode_verified_frames(pixels(), tmp_path / "video", terminal_tick=4)
    assert result["master_verified"]["frames"] == 5
    assert result["master_verified"]["rgb_hashes_match"] is True
    assert result["master_verified"]["native_crop_hashes_match"] is True
    assert result["view_verified"]["frames"] == 5
    assert result["view_verified"]["displayed_duration_s"] == 0.05
    assert result["view_verified"]["physics_duration_s"] == 0.04
    assert result["view_verified"]["final_terminal_display_s"] == 0.01
    assert result["human_review_performed"] is False
    assert not list((tmp_path / "video").glob("*.png"))


def test_encoder_rejects_hud_touching_native_pixels(tmp_path):
    m = module()
    frames = list(pixels())
    frames[2][1][0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="native"):
        m.encode_verified_frames(iter(frames), tmp_path / "video", terminal_tick=4)
    assert (tmp_path / "video" / "failure.json").is_file()


def test_encoder_rejects_omitted_or_duplicate_time_and_never_overwrites(tmp_path):
    m = module()
    frames = list(pixels())
    frames[2][2]["index"] = 3
    with pytest.raises(ValueError, match="timestamp"):
        m.encode_verified_frames(iter(frames), tmp_path / "video", terminal_tick=4)
    with pytest.raises(FileExistsError):
        m.encode_verified_frames(pixels(), tmp_path / "video", terminal_tick=4)


def test_decoded_master_check_catches_changed_expected_rgb_hash(tmp_path):
    m = module()
    path = tmp_path / "video"
    m.encode_verified_frames(pixels(), path, terminal_tick=4)
    rows = [json.loads(line) for line in (path / "frame-ledger.jsonl").read_text().splitlines()]
    rows[3]["hud_rgb_sha256"] = "0" * 64
    (path / "frame-ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="RGB"):
        m.verify_video_files(path, terminal_tick=4, shape=(16, 24, 3), native_height=12)


def test_final_runner_refuses_noncapability_before_files_or_native_import(tmp_path):
    m = module()
    for capability in [None, True, {"selected_seed": 42}]:
        with pytest.raises(ValueError, match="VerifiedJointFreeze"):
            m.run_final_media(freeze=capability, original="missing", font="missing", output=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_prospective_exact_video_ids_reject_missing_reordered_or_heldout_cases():
    m = module()
    from adaptive_drone_patrol.dual_coordination_cases import generate_case

    ids = [
        "development-canonical-normal",
        "development-video-priority-transfer",
        "development-video-crossing-yield",
    ]
    rows = []
    for index, identifier in enumerate(ids):
        case = generate_case(15, index, "priority" if index == 1 else "normal")
        case["case_id"] = identifier
        rows.append(dict(case=case))
    declaration = dict(schema="dual-campaign-declaration-v1", video_case_ids=ids)
    assert list(m.declared_video_cases(declaration, rows)) == list(m.ROLES)
    with pytest.raises(ValueError, match="IDs"):
        m.declared_video_cases(dict(declaration, video_case_ids=list(reversed(ids))), rows)
    with pytest.raises(ValueError, match="missing"):
        m.declared_video_cases(declaration, rows[:2])
    rows[0]["case"]["split"] = 30
    with pytest.raises(ValueError, match="development"):
        m.declared_video_cases(declaration, rows)


def test_transfer_metric_requires_actual_post_event_new_owner_success():
    m = module()
    state = dict(
        tick=200,
        terminal_reason="success",
        completed_counts=[5, 5],
        final_dwell_ticks=[100, 100],
        goals=[dict(status="DONE", owner=i % 2, completed_tick=180) for i in range(10)],
        event_receipts=[dict(observed_tick=100, no_op=False)],
        action_receipts=[dict(tick=0, owners=[0] + [None] * 9), dict(tick=120, owners=[1] + [None] * 9)],
    )
    state["goals"][0]["owner"] = 1
    audit = dict(safe=True, violation_count=0, minimum_clearance_m=0.8, floor_minima_m=[0.5, 0.5])
    metrics = m.behavior_metrics(dict(case_id="synthetic"), state, audit)
    assert metrics["successful_post_event_transfers"] == 1
    state["action_receipts"][1]["tick"] = 80
    assert m.behavior_metrics(dict(case_id="synthetic"), state, audit)["successful_post_event_transfers"] == 0
    state["action_receipts"][1]["tick"] = 120
    state["goals"][0]["owner"] = 0
    assert m.behavior_metrics(dict(case_id="synthetic"), state, audit)["successful_post_event_transfers"] == 0


def test_source_trace_rejects_sparse_trace_without_reading_or_interpolating_it(tmp_path):
    m = module()
    trace = tmp_path / "replay-0000.draw"
    trace.write_bytes(b"not used because sparse")
    (tmp_path / "environment.json").write_text(json.dumps(dict(case=dict(split=15))))
    result = dict(
        mission=dict(state=dict(tick=20)),
        replay_segments=[
            dict(
                path=str(trace),
                bytes=trace.stat().st_size,
                sha256=m.sha256(trace),
                start_tick=0,
                end_tick=20,
                records=3,
            )
        ],
    )
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="sparse"):
        m._source_trace(tmp_path)


def test_decoder_checks_actual_frame_dimensions(tmp_path):
    m = module()
    path = tmp_path / "video"
    m.encode_verified_frames(pixels(), path, terminal_tick=4)
    log = path / "decode-master.log"
    with pytest.raises(ValueError, match="dimensions"):
        m._timeline(log, 4, shape=(24, 16, 3))


def test_retire_trace_preserves_inputs_and_refuses_changed_reproduction_or_video(tmp_path):
    m = module()
    slot = tmp_path / "synthetic-owned-slot"
    source = slot / "capture"
    source.mkdir(parents=True)
    render = slot / "render"
    encoding = m.encode_verified_frames(pixels(), render / "video", terminal_tick=4)
    (render / "render-verification.json").write_text(
        json.dumps(
            dict(
                full_mission_machine_checked=True,
                frame_coverage=dict(every_frame_verified=True),
                encoding=encoding,
            )
        )
    )
    (slot / "media-owned-slot.json").write_text(
        json.dumps(dict(schema="dual-media-owned-slot-v1", freeze_sha256="test"))
    )
    for name in (
        "reset-inputs.json",
        "proposed-actions.jsonl",
        "capture.json",
        "environment.json",
        "result.json",
    ):
        (source / name).write_text("{}")
    trace = source / "replay-0000.draw"
    trace.write_bytes(b"synthetic-retention-contract-only")
    verification = dict(
        full_rate_inputs_verified=True,
        trace=m.file_record(trace, source),
        reproduction_inputs={p.name: m.file_record(p, source) for p in source.glob("*.json*")},
    )
    (source / "reset-inputs.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Reproduction inputs changed"):
        m.retire_verified_render_trace(source, render, verification, freeze_digest="test")
    assert trace.exists()
    (source / "reset-inputs.json").write_text("{}")
    view = render / "video" / "view.mp4"
    original = view.read_bytes()
    view.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="media or frame ledger changed"):
        m.retire_verified_render_trace(source, render, verification, freeze_digest="test")
    assert trace.exists()
    view.write_bytes(original)
    result = m.retire_verified_render_trace(source, render, verification, freeze_digest="test")
    assert result["retired"] is True and not trace.exists()
    assert all((source / name).is_file() for name in verification["reproduction_inputs"])


def test_full_development_native_render_when_explicitly_requested(tmp_path):
    import os

    if os.environ.get("DUAL_MEDIA_FULL") != "1":
        pytest.skip("Full native media requires explicit pilot/original/Japanese-font inputs")
    pytest.importorskip("mujoco")
    m = module()
    result = m.render_development_video(
        os.environ["DUAL_RENDER_PILOT"],
        os.environ["DUAL_CF2X"],
        tmp_path / "full-media",
        os.environ["DUAL_FONT"],
    )
    assert result["scope"] == "DEVELOPMENT_ONLY"
    assert result["frame_coverage"]["every_frame_verified"] is True
    assert result["encoding"]["master_verified"]["native_crop_hashes_match"] is True
    assert result["human_review_performed"] is False


def runtime_contract_pair():
    """Saved JSON arrays versus runtime()'s actual tuple exclusion representation."""
    from copy import deepcopy

    recorded = dict(
        schema="dual-native-properties-v1",
        version="3.15.0",
        precision="float64",
        model_xml_sha256="a" * 64,
        original_asset_sha256="c" * 64,
        body_mass=[0.0, 0.027000000700354576],
        geom_size=[[0.03, 0.03, 0.03]],
        body_ids=[dict(base_link=1)],
        self_exclusions=[["drone-1/base_link", "drone-1/suspension_rod"]],
    )
    current = deepcopy(recorded)
    current["model_xml_sha256"] = "b" * 64
    current["self_exclusions"] = [tuple(pair) for pair in recorded["self_exclusions"]]
    return recorded, current


def test_native_runtime_json_tuple_array_difference_preserves_strict_physical_identity():
    m = module()
    recorded, current = runtime_contract_pair()
    assert recorded["self_exclusions"] != current["self_exclusions"]
    proof = m.verify_runtime_identity(recorded, current)
    assert proof["exact_physical_properties_match"] is True
    assert proof["recorded_model_xml_sha256"] == "a" * 64
    assert proof["render_model_xml_sha256"] == "b" * 64
    assert current["self_exclusions"][0] == ("drone-1/base_link", "drone-1/suspension_rod")


@pytest.mark.parametrize(
    "field", ["body_mass", "geom_size", "self_exclusions", "body_ids", "original_asset_sha256", "version"]
)
def test_native_runtime_normalization_cannot_hide_real_value_or_identity_changes(field):
    m = module()
    recorded, current = runtime_contract_pair()
    if field == "body_mass":
        current[field][1] = float(np.nextafter(current[field][1], np.inf))
    elif field == "geom_size":
        current[field][0][0] = 0.030000000000000002
    elif field == "self_exclusions":
        current[field][0] = ("drone-1/base_link", "drone-1/payload")
    elif field == "body_ids":
        current[field][0]["base_link"] = 2
    else:
        current[field] = "changed"
    with pytest.raises(ValueError, match=field):
        m.verify_runtime_identity(recorded, current)


def test_native_runtime_normalization_rejects_missing_extra_and_nonfinite_properties():
    m = module()
    recorded, current = runtime_contract_pair()
    current["new_unmatched_field"] = 1
    with pytest.raises(ValueError, match="new_unmatched_field"):
        m.verify_runtime_identity(recorded, current)
    del current["new_unmatched_field"]
    del current["body_ids"]
    with pytest.raises(ValueError, match="body_ids"):
        m.verify_runtime_identity(recorded, current)
    current["body_ids"] = recorded["body_ids"]
    current["body_mass"][1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.verify_runtime_identity(recorded, current)
