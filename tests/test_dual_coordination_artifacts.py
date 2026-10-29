"""Real-file artifact preservation and exact-final-media review contracts."""
from copy import deepcopy
import hashlib

import pytest

from adaptive_drone_patrol.dual_coordination_artifacts import (
    compare_inventories, inventory, recover_local, verify_human_reviews,
)


def make_source(root):
    root.mkdir()
    (root/'nested').mkdir()
    (root/'nested'/'payload.bin').write_bytes(b'abc')
    (root/'empty.txt').write_bytes(b'')
    return root


def test_inventory_hashes_all_bytes_and_orders_portable_relative_paths(tmp_path):
    source = make_source(tmp_path/'source')
    result = inventory(source)
    assert list(result['files']) == ['empty.txt','nested/payload.bin']
    assert result['files']['nested/payload.bin'] == {
        'bytes':3, 'sha256':'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    }
    assert result['files']['empty.txt']['sha256'] == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    assert result['total_bytes'] == 3
    assert result['file_count'] == 2


def test_inventory_rejects_root_file_missing_root_and_any_symlink(tmp_path):
    source = make_source(tmp_path/'source')
    with pytest.raises(ValueError):
        inventory(tmp_path/'absent')
    with pytest.raises(ValueError):
        inventory(source/'empty.txt')
    root_link = tmp_path/'root-link'
    root_link.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError):
        inventory(root_link)
    for target in (source/'empty.txt',source/'nested',tmp_path/'missing'):
        link = source/'link'
        link.symlink_to(target)
        with pytest.raises(ValueError):
            inventory(source)
        link.unlink()


def test_exclusions_are_explicit_files_not_patterns_or_directory_subtrees(tmp_path):
    source = make_source(tmp_path/'source')
    (source/'inventory.json').write_text('old manifest')
    result = inventory(source,exclude=('inventory.json',))
    assert result['excluded'] == ['inventory.json']
    assert result['file_count'] == 2
    (source/'inventory.json').write_text('new manifest describing this inventory')
    assert inventory(source,exclude=('inventory.json',)) == result
    for excluded in ('nested','../outside','/absolute','*.json','nested/../empty.txt','./empty.txt'):
        with pytest.raises(ValueError):
            inventory(source,exclude=(excluded,))


def test_excluded_symlink_is_still_rejected(tmp_path):
    source = make_source(tmp_path/'source')
    (source/'inventory.json').symlink_to(source/'empty.txt')
    with pytest.raises(ValueError):
        inventory(source,exclude=('inventory.json',))


def test_comparison_reports_missing_extra_size_and_hash_changes(tmp_path):
    source = make_source(tmp_path/'source')
    expected = inventory(source)
    assert compare_inventories(expected,deepcopy(expected))['verified'] is True
    missing = deepcopy(expected)
    del missing['files']['empty.txt']
    missing['file_count'] -= 1
    with pytest.raises(ValueError,match='missing'):
        compare_inventories(expected,missing)
    extra = deepcopy(expected)
    extra['files']['extra.txt'] = extra['files']['empty.txt']
    extra['file_count'] += 1
    with pytest.raises(ValueError,match='extra'):
        compare_inventories(expected,extra)
    changed = deepcopy(expected)
    changed['files']['nested/payload.bin']['sha256'] = '0'*64
    with pytest.raises(ValueError,match='mismatch'):
        compare_inventories(expected,changed)
    changed['files']['nested/payload.bin']['bytes'] = 4
    changed['total_bytes'] = 4
    with pytest.raises(ValueError,match='mismatch'):
        compare_inventories(expected,changed)


@pytest.mark.parametrize('unsafe',['../escape','/absolute','./relative','nested//file','C:\\outside','nested/../file'])
def test_manifest_paths_cannot_escape_or_alias(tmp_path,unsafe):
    source = make_source(tmp_path/'source')
    expected = inventory(source)
    malformed = deepcopy(expected)
    malformed['files'][unsafe] = malformed['files'].pop('nested/payload.bin')
    with pytest.raises(ValueError):
        compare_inventories(malformed,malformed)


@pytest.mark.parametrize('field,value',[('bytes',True),('bytes',-1),('bytes',1.1),('sha256','x'*64),('sha256','0'*63)])
def test_manifest_file_records_reject_invalid_sizes_and_hashes(tmp_path,field,value):
    source = make_source(tmp_path/'source')
    malformed = inventory(source)
    malformed['files']['nested/payload.bin'][field] = value
    with pytest.raises(ValueError):
        compare_inventories(malformed,malformed)


def test_local_recovery_creates_new_copy_independently_rehashes_and_retains_source(tmp_path):
    source = make_source(tmp_path/'source')
    before = inventory(source)
    destination = tmp_path/'recovered'
    receipt = recover_local(source,destination)
    assert receipt['verified'] is True
    assert receipt['transport'] == 'local-filesystem-copy'
    assert receipt['remote_transfer_verified'] is False
    assert receipt['source_inventory'] == before
    assert receipt['destination_inventory'] == before
    assert inventory(source) == before
    assert inventory(destination) == before
    (destination/'nested'/'payload.bin').write_bytes(b'xyz')
    with pytest.raises(ValueError,match='mismatch'):
        compare_inventories(inventory(source),inventory(destination))
    assert (source/'nested'/'payload.bin').read_bytes() == b'abc'


def test_recovery_rejects_existing_or_nested_destination_without_mutating_source(tmp_path):
    source = make_source(tmp_path/'source')
    before = inventory(source)
    existing = tmp_path/'existing'
    existing.mkdir()
    (existing/'keep').write_text('untouched')
    for destination in (existing,source/'nested'/'recovered',source):
        with pytest.raises(ValueError):
            recover_local(source,destination)
    assert (existing/'keep').read_text() == 'untouched'
    assert inventory(source) == before


def test_recovery_exclusion_is_declared_and_not_copied(tmp_path):
    source = make_source(tmp_path/'source')
    (source/'inventory.json').write_text('source inventory excluded explicitly')
    receipt = recover_local(source,tmp_path/'recovered',exclude=('inventory.json',))
    assert receipt['verified']
    assert receipt['source_inventory']['excluded'] == ['inventory.json']
    assert not (tmp_path/'recovered'/'inventory.json').exists()
    assert (source/'inventory.json').exists()


def media_and_reviews(root):
    videos = [dict(role=role,path=f'videos/{role}.mp4',bytes=100,
                   sha256=hashlib.sha256(role.encode()).hexdigest())
              for role in ('normal','priority-transfer','crossing-yield')]
    (root/'videos').mkdir(exist_ok=True)
    for video in videos:
        content = video['role'].encode()
        (root/video['path']).write_bytes(content)
        video['bytes'] = len(content)
    media = dict(schema='dual-media-v1',finalized_at='2026-10-27T15:00:00Z',videos=videos)
    reviews = [dict(path=v['path'],sha256=v['sha256'],reviewer='human-reviewer',
                    reviewed_at='2026-10-27T15:30:00+00:00',full_length=True,
                    playback_speed=1.0,approved=True) for v in videos]
    return media,reviews


def test_human_review_requires_exact_three_final_hashes_and_fresh_full_length_1x_receipts(tmp_path):
    manifest,receipts = media_and_reviews(tmp_path)
    result = verify_human_reviews(manifest,receipts,media_root=tmp_path)
    assert result['verified'] is True
    assert result['reviewed_files'] == 3
    assert result['machine_verified_media'] is False


@pytest.mark.parametrize('field,value',[
    ('sha256','0'*64),('reviewed_at','2026-10-26T15:30:00Z'),
    ('reviewed_at','2026-10-27T15:30:00'),('reviewed_at','not a date'),
    ('full_length',False),('full_length',1),('playback_speed',2),
    ('playback_speed',True),('approved',False),('reviewer',' '),
])
def test_human_review_rejects_stale_partial_accelerated_or_unidentified_review(tmp_path,field,value):
    manifest,receipts = media_and_reviews(tmp_path)
    receipts[0][field] = value
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)


def test_human_review_cannot_reuse_one_receipt_or_skip_one_video(tmp_path):
    manifest,receipts = media_and_reviews(tmp_path)
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts[:2],media_root=tmp_path)
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts+[receipts[0]],media_root=tmp_path)
    receipts[2] = receipts[0]
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)


def test_media_manifest_rejects_wrong_schema_duplicate_roles_and_unsafe_paths(tmp_path):
    manifest,receipts = media_and_reviews(tmp_path)
    for changed in (dict(manifest,schema='old-single-drone'),dict(manifest,videos=manifest['videos'][:2])):
        with pytest.raises(ValueError):
            verify_human_reviews(changed,receipts,media_root=tmp_path)
    manifest['videos'][1]['role'] = 'normal'
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)
    manifest,receipts = media_and_reviews(tmp_path)
    manifest['videos'][0]['path'] = '../old.mp4'
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)


def test_final_review_rehashes_actual_media_and_rejects_missing_changed_or_linked_files(tmp_path):
    manifest,receipts = media_and_reviews(tmp_path)
    final = tmp_path/manifest['videos'][0]['path']
    final.write_bytes(b'edited after review')
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)
    final.unlink()
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)
    final.symlink_to(tmp_path/manifest['videos'][1]['path'])
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)


def test_future_dated_review_cannot_certify_an_unperformed_human_review(tmp_path):
    manifest,receipts = media_and_reviews(tmp_path)
    receipts[0]['reviewed_at'] = '2999-01-01T00:00:00Z'
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)


def test_malformed_media_role_is_rejected_as_a_contract_error(tmp_path):
    manifest,receipts = media_and_reviews(tmp_path)
    manifest['videos'][0]['role'] = ['normal']
    with pytest.raises(ValueError):
        verify_human_reviews(manifest,receipts,media_root=tmp_path)


def test_manifest_cannot_declare_an_ancestor_directory_as_excluded_file(tmp_path):
    source = make_source(tmp_path/'source')
    malformed = inventory(source)
    malformed['excluded'] = ['nested']
    with pytest.raises(ValueError):
        compare_inventories(malformed,malformed)


def test_special_file_is_rejected_without_opening_or_hanging(tmp_path):
    import os
    source = make_source(tmp_path/'source')
    os.mkfifo(source/'pipe')
    with pytest.raises(ValueError):
        inventory(source)


def test_recovery_materializes_exclusion_iterators_for_every_independent_inventory(tmp_path):
    source = make_source(tmp_path/'source')
    (source/'inventory.json').write_text('excluded manifest')
    receipt = recover_local(source,tmp_path/'destination',exclude=iter(['inventory.json']))
    assert receipt['verified'] is True
    assert receipt['source_inventory']['excluded'] == ['inventory.json']
    assert receipt['destination_inventory']['excluded'] == ['inventory.json']
