import pytest
from adaptive_drone_patrol.elevated_artifacts import inventory,verify_inventory


def test_recovered_size_and_sha_detect_same_length_corruption(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'raw.bin').write_bytes(b'raw1')
    expected=inventory(source)
    dest=tmp_path/'dest';dest.mkdir();(dest/'raw.bin').write_bytes(b'raw1')
    assert verify_inventory(dest,expected)['all_verified']
    (dest/'raw.bin').write_bytes(b'raw2')
    with pytest.raises(ValueError,match='mismatch'):verify_inventory(dest,expected)
    (dest/'raw.bin').write_bytes(b'raw-longer')
    with pytest.raises(ValueError):verify_inventory(dest,expected)
    expected['files']['../outside']={'bytes':0,'sha256':'0'*64}
    with pytest.raises(ValueError):verify_inventory(dest,expected)


def test_recovery_inventory_includes_declared_ram_directory_symlink(tmp_path):
    source=tmp_path/'source';source.mkdir();ram=tmp_path/'ram';ram.mkdir();(ram/'stage.npz').write_bytes(b'actual-stage')
    (source/'basic-normal-1').symlink_to(ram,target_is_directory=True)
    expected=inventory(source)
    assert expected['file_count']==1
    recovered=tmp_path/'recovered';(recovered/'basic-normal-1').mkdir(parents=True)
    (recovered/'basic-normal-1/stage.npz').write_bytes(b'actual-stage')
    assert verify_inventory(recovered,expected)['all_verified']


def test_inventory_rejects_missing_roots_and_dangling_ram_links(tmp_path):
    with pytest.raises(ValueError):inventory(tmp_path/'missing')
    root=tmp_path/'root';root.mkdir()
    (root/'basic-normal-1').symlink_to(tmp_path/'lost-ram',target_is_directory=True)
    with pytest.raises(ValueError,match='Dangling'):inventory(root)


def test_inventory_rejects_traversal_errors(tmp_path,monkeypatch):
    import os
    def denied(*args,**kwargs):
        kwargs['onerror'](PermissionError('unreadable subtree'))
        return iter(())
    monkeypatch.setattr(os,'walk',denied)
    with pytest.raises(PermissionError):inventory(tmp_path)


def test_inventory_rejects_missing_declared_case_subtree(tmp_path):
    (tmp_path/'declaration.json').write_text('{"cases":[{"case_id":"basic-normal-1"}]}')
    with pytest.raises(ValueError,match='declared'):inventory(tmp_path)


def test_inventory_rejects_nested_missing_declared_case(tmp_path):
    formal=tmp_path/'formal';formal.mkdir()
    (formal/'declaration.json').write_text('{"cases":[{"case_id":"basic-normal-1"}]}')
    with pytest.raises(ValueError,match='declared'):inventory(tmp_path)


def test_bound_audit_rejects_changed_control_and_missing_other_case(tmp_path):
    from adaptive_drone_patrol.elevated_campaign import audit_source_hashes,validate_audit_receipt
    root=tmp_path/'case';root.mkdir();(root/'control.json').write_text('old')
    proof=dict(case_id='case',passed=True,artifacts=inventory(root),checked_sources=audit_source_hashes())
    validate_audit_receipt(root,dict(case_id='case'),proof)
    (root/'control.json').write_text('new')
    with pytest.raises(ValueError):validate_audit_receipt(root,dict(case_id='case'),proof)
