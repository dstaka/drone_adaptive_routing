"""Admission must replay current raw evidence, never consume a status as proof."""
import copy
import importlib
from pathlib import Path
import pytest


def api():
    return importlib.import_module('adaptive_drone_patrol.payload_retry_admission')


def test_old_pass_cannot_admit_without_complete_replay():
    with pytest.raises(ValueError):
        api().require_retry_admission({'status':'PASS','physical_training_admitted':True},{},{'status':'PASS'})


def test_current_receipt_is_replayed_and_current_source_is_required(tmp_path, monkeypatch):
    module=api()
    root=Path.cwd()
    actual={'profile':'cf2x-two-native-ball-root-aligned-v3','source_binding':{'physics.py':'a'*64}}
    receipt=dict(schema='payload-operating-domain-independent-v3',status='QUALIFIED_FOR_TRAINING',
        physical_training_admitted=True,selected_profile=actual['profile'],failure_codes=[],
        source_binding={'physics.py':'a'*64},physical_signature_sha256='b'*64,
        checklist={'all':{'passed':True}})
    locations=dict(schema='payload-retry-evidence-locations-v1',probes='p',motor_probes='m',reset_probe='r',initial_cases='c')
    calls=[]
    monkeypatch.setattr(module,'installed_engine_root',lambda:tmp_path)
    def replay(repository, values, engine):
        calls.append((repository,values,engine))
        return copy.deepcopy(receipt)
    monkeypatch.setattr(module,'replay_current_evidence',replay)
    assert module.require_retry_admission(receipt,actual,locations)['physical_signature_sha256']=='b'*64
    assert calls==[(root,locations,tmp_path)]
    damaged=copy.deepcopy(receipt)
    damaged['physical_signature_sha256']='c'*64
    with pytest.raises(ValueError):
        module.require_retry_admission(damaged,actual,locations)
    actual['source_binding']['physics.py']='d'*64
    with pytest.raises(ValueError):
        module.require_retry_admission(receipt,actual,locations)
