"""Narrow explicit new-profile bindings, separate from immutable old readers."""
import copy,json
from pathlib import Path
from .analytical import content_hash
from .elevated_layout import PROFILE,elevated_checks
from .elevated_media import load_elevated_camera
from .util import sha256,load_config
from .scenario import Scenario

LAYOUT=Path('configs/demo_scenarios_payload_elevated/layout.json')


def development_binding(routing,camera,*,selection=None,provisional=False):
    routing,camera=Path(routing),Path(camera)
    actual=sha256(routing)
    if provisional:
        from .elevated_routing_experiment import BASELINE_SHA256
        if selection is not None or actual!=BASELINE_SHA256:raise ValueError('Only explicit baseline provisional development allowed')
        model_name='B'; selected_hash=None
    else:
        if selection is None:raise ValueError('Validation selection receipt required')
        selected=json.loads(Path(selection).read_text()); model_name=selected['selected']
        if not model_name or not selected['eligible'][model_name] or selected['models'][model_name]['sha256']!=actual:
            raise ValueError('Selected checkpoint differs')
        selected_hash=sha256(selection)
    load_elevated_camera(camera)
    layout=json.loads(LAYOUT.read_text())
    if layout['profile']!=PROFILE or not elevated_checks(layout['coords'])['passed']:
        raise ValueError('Invalid elevated layout')
    for name,item in layout['scenarios'].items():
        if sha256(LAYOUT.parent/f'{name}.yaml')!=item['sha256']:raise ValueError('Scenario file differs')
    return dict(schema='payload-elevated-binding-v1',profile=PROFILE,layout_sha256=sha256(LAYOUT),
        scenarios=copy.deepcopy(layout['scenarios']),camera_path=str(camera),camera_sha256=sha256(camera),
        routing_path=str(routing),routing_sha256=actual,routing_schema_sha256=sha256(routing.with_suffix('.schema.json')),
        routing_model_name=model_name,selection_path=str(selection) if selection else None,selection_sha256=selected_hash,
        provisional=provisional,controller='C-local-lqr-v1',flight_nn='NOT_USED')


def verify_binding_scenario(scenario,case,binding):
    expected=development_binding(binding['routing_path'],binding['camera_path'],
                                 selection=binding['selection_path'],provisional=binding['provisional'])
    if binding!=expected:raise ValueError('Current profile/model/camera binding differs')
    name=case['scenario']
    if name not in binding['scenarios']:raise ValueError('Unknown elevated scenario')
    expected=Scenario.from_dict(load_config(LAYOUT.parent/f'{name}.yaml'))
    if scenario.to_dict()!=expected.to_dict():raise ValueError('Mixed or stale elevated scenario')
    return True
