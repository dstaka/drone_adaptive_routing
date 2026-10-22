"""Reconstruct complete current physical evidence before any simulator import."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def installed_engine_root():
    """Locate installed bytes without importing Genesis or initializing a GPU."""
    if importlib.metadata.version('genesis-world') != '1.4.3':
        raise ValueError('Physical admission requires pinned Genesis1.4.3')
    spec=importlib.util.find_spec('genesis')
    if spec is None or spec.origin is None:
        raise ValueError('Installed engine source is unavailable')
    return Path(spec.origin).parent


def replay_current_evidence(root, locations, engine_root):
    path=root/'scripts/check_payload_retry_qualification.py'
    spec=importlib.util.spec_from_file_location('retry_admission_replay',path)
    checker=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker.verify_qualification(root/locations['probes'],root/locations['motor_probes'],
        root/locations['reset_probe'],source_root=root,initial_cases=root/locations['initial_cases'],
        engine_root=engine_root,geometry_root=root)


def require_retry_admission(receipt, actual_binding, locations):
    """A claimed qualified status is necessary but never sufficient for replay."""
    if (receipt.get('schema')!='payload-operating-domain-independent-v3' or
            receipt.get('status')!='QUALIFIED_FOR_TRAINING' or receipt.get('physical_training_admitted') is not True or
            receipt.get('failure_codes')!=[] or
            receipt.get('selected_profile')!=actual_binding.get('profile') or
            actual_binding.get('profile')!='cf2x-two-native-ball-root-aligned-v3'):
        raise ValueError('Retry diagnostics-only: complete new qualification is required')
    if (not receipt.get('source_binding') or any(actual_binding.get('source_binding',{}).get(name)!=wanted
            for name,wanted in receipt['source_binding'].items()) or
            not receipt.get('checklist') or any(stage.get('passed') is not True for stage in receipt['checklist'].values())):
        raise ValueError('Actual physics source or required qualification stage differs')
    if (locations.get('schema')!='payload-retry-evidence-locations-v1' or
            any(not isinstance(locations.get(key),str) or not locations[key]
                for key in ('probes','motor_probes','reset_probe','initial_cases'))):
        raise ValueError('Actual raw evidence locations are required; a reconstruction status is insufficient')
    replay=replay_current_evidence(Path.cwd(),locations,installed_engine_root())
    if (replay.get('physical_training_admitted') is not True or
            json.dumps(replay,sort_keys=True,allow_nan=False)!=json.dumps(receipt,sort_keys=True,allow_nan=False)):
        raise ValueError('Current independent raw reconstruction differs from the qualification receipt')
    return dict(physical_signature_sha256=replay['physical_signature_sha256'],
                current_raw_evidence_replayed=True)
