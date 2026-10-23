"""Actual lossless codec round trips must retain every RGB frame and100Hz PTS."""

import json
import numpy as np
import pytest


def test_real_lossless_pair_preserves_raw_view_and_all_pts(tmp_path):
    from adaptive_drone_patrol.payload_solver_video import PairedVideoWriter, verify_video_pair

    rng = np.random.default_rng(42)
    expected = []
    with PairedVideoWriter(tmp_path / 'video', width=64, raw_height=32, hud_height=16) as writer:
        for step in range(4):
            raw = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
            hud = np.concatenate([raw, np.full((16, 64, 3), step * 30, dtype=np.uint8)])
            writer.append(step, raw, hud)
            expected.append(raw)
    proof = verify_video_pair(tmp_path / 'video')
    assert proof['frames'] == 4 and proof['fps'] == 100
    assert proof['last_physical_time_s'] == .03
    assert proof['all_lossless_pixels_match'] is True
    assert proof['raw_viewport_equal_every_frame'] is True
    assert proof['pts'] == [0, .01, .02, .03]


def test_video_writer_rejects_repeated_event_frame_and_changed_raw_pixels(tmp_path):
    from adaptive_drone_patrol.payload_solver_video import PairedVideoWriter

    raw = np.zeros((32, 64, 3), dtype=np.uint8)
    hud = np.zeros((48, 64, 3), dtype=np.uint8)
    with PairedVideoWriter(tmp_path / 'video', width=64, raw_height=32, hud_height=16) as writer:
        writer.append(0, raw, hud)
        with pytest.raises(ValueError, match='step'):
            writer.append(0, raw, hud)
        hud[0, 0, 0] = 1
        with pytest.raises(ValueError, match='raw'):
            writer.append(1, raw, hud)


def test_decoded_rgb_hash_mismatch_cannot_be_approved(tmp_path):
    from adaptive_drone_patrol.payload_solver_video import PairedVideoWriter, verify_video_pair

    raw = np.zeros((32, 64, 3), dtype=np.uint8)
    hud = np.zeros((48, 64, 3), dtype=np.uint8)
    root = tmp_path / 'video'
    with PairedVideoWriter(root, width=64, raw_height=32, hud_height=16) as writer:
        writer.append(0, raw, hud)
    path = root / 'frames.jsonl'
    row = json.loads(path.read_text())
    row['raw_rgb_sha256'] = 'a' * 64
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='RGB'):
        verify_video_pair(root)


def test_real_viewing_copies_decode_every_frame_at_same_physical_times(tmp_path):
    from adaptive_drone_patrol.payload_solver_video import (
        PairedVideoWriter, create_viewing_copies, verify_viewing_copies,
    )

    raw = np.repeat(np.arange(64, dtype=np.uint8)[None, :, None], 32, axis=0)
    raw = np.repeat(raw, 3, axis=2)
    root = tmp_path / 'video'
    with PairedVideoWriter(root, width=64, raw_height=32, hud_height=16) as writer:
        for step in range(3):
            writer.append(step, raw + step, np.concatenate([raw + step, np.zeros((16, 64, 3), np.uint8)]))
    create_viewing_copies(root)
    proof = verify_viewing_copies(root)
    assert proof['raw']['frames'] == proof['hud']['frames'] == 3
    assert proof['raw']['maximum_frame_mean_absolute_rgb_error'] < 6
    assert proof['hud']['pts_s'] == [0, .01, .02]


def test_erasing_small_caption_cannot_hide_in_whole_frame_average():
    from PIL import Image, ImageDraw, ImageFont
    from adaptive_drone_patrol.payload_solver_video import verify_caption_regions

    im = Image.new('RGB',(1792,1200),'#101d2d')
    ImageDraw.Draw(im).text((22,780),'Clock 12.34 altitude 1.002',fill='white',font=ImageFont.load_default(size=24))
    original = np.asarray(im)
    changed = original.copy()
    changed[780:810,22:220] = [16,29,45]
    assert np.abs(original.astype(float)-changed).mean() < 6
    with pytest.raises(ValueError,match='caption'):
        verify_caption_regions(original,changed,{'clock': [20,775,800,825]})
    proof = verify_caption_regions(original,original,{'clock': [20,775,800,825]})
    assert proof['maximum_region_luminance_mae'] == 0
    assert proof['maximum_foreground_tile_luminance_mae'] == 0
