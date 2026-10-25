"""Read-only reconciliation of policy/derived samples against native raw."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np

from .payload_contract import rotation
from .payload_retry_contract import actor_observation
from .payload_solver_policy import mean_policy
from .payload_solver_qualification import physical_errors
from .util import sha256


def observation_sample(raw, side):
    sample = copy.deepcopy(raw)
    for key, body, offset in [("a", "drone", [0, 0, -0.02]), ("b", "package", [0, 0, side / 2])]:
        arm = rotation(raw[body + "_quaternion_wxyz"]) @ offset
        sample["attachment_" + key + "_world"] = (np.array(raw[body + "_xyz"]) + arm).tolist()
        sample["attachment_" + key + "_velocity"] = (
            np.array(raw[body + "_velocity"]) + np.cross(raw[body + "_angular_velocity"], arm)
        ).tolist()
    return sample



def match_native_sample(sample, actual, properties):
    """Require the complete qualified native schema and a single contact truth."""
    qwidth = {0: 7, 1: 4, 2: 1, 3: 1}
    vwidth = {0: 6, 1: 3, 2: 1, 3: 1}
    try:
        nq = sum(qwidth[k] for k in properties['joint_type'])
        nv = sum(vwidth[k] for k in properties['joint_type'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Unknown native model joint schema') from exc
    shapes = dict(qpos=(nq,), qvel=(nv,), qacc=(nv,), rpm=(4,),
        actual_applied_generalized_force=(nv,), actual_passive_generalized_force=(nv,),
        actual_xfrc_applied=(len(properties['links']) + 1, 6))
    for body in ['drone', 'rod', 'package']:
        for field, width in [('xyz',3), ('velocity',3), ('angular_velocity',3), ('quaternion_wxyz',4)]:
            shapes[body + '_' + field] = (width,)
    required = set(shapes) | {'time_s', 'contact', 'warning_counts'}
    if set(actual) != required:
        raise ValueError('Incomplete or unknown actual native getter schema')
    for key, shape in shapes.items():
        values = np.asarray(actual[key], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Invalid actual native field: ' + key)
    if (type(actual['contact']) is not bool or type(sample.get('contacts')) is not bool
            or sample['contacts'] != actual['contact']):
        raise ValueError('Native contact and safety contact alias differ')
    warnings = actual['warning_counts']
    if len(warnings) != 7 or any(type(w) is not int or w != 0 for w in warnings):
        raise ValueError('Native solver warnings or missing warning inventory')
    if (type(actual['time_s']) not in (int,float) or not np.isfinite(actual['time_s'])
            or actual['time_s'] < 0):
        raise ValueError('Invalid native physical clock')
    for body in ['drone', 'rod', 'package']:
        rotation(actual[body + '_quaternion_wxyz'])
        if (np.linalg.norm(actual[body + '_velocity']) > 20
                or np.linalg.norm(actual[body + '_angular_velocity']) > 200):
            raise ValueError('Native numerical speed excursion')
    for key, value in actual.items():
        if key == 'time_s':
            if not np.isclose(sample.get(key, float('nan')), value, atol=1e-7, rtol=0):
                raise ValueError('Native engine clock differs')
        elif key not in sample or not np.array_equal(sample[key], value):
            raise ValueError('Derived telemetry differs from native getter: ' + key)
    return True


def verify_native_sequence(rows, native_path, properties_path, checkpoint, expected_hash, case, binding):
    """Reconstruct every actor input/output and reconcile all actual getter fields."""
    import torch
    from .payload_solver_validation import verify_initial_sample
    from .payload_retry_contract import OBSERVATION_CONTRACT

    native_path, properties_path = Path(native_path), Path(properties_path)
    if sha256(checkpoint) != expected_hash:
        raise ValueError("Frozen actual policy artifact differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = state["infos"]
    if (
        info.get("actual_solver_binding") != binding
        or info.get("observation_contract") != OBSERVATION_CONTRACT
    ):
        raise ValueError("Actual raw actor/physics binding differs")
    properties = json.loads(properties_path.read_text())
    if physical_errors(
        properties, dict(side_m=case["side_m"], mass_ratio=case["mass_ratio"], substeps=16), fixed=False
    ):
        raise ValueError("Actual recorded native properties differ")
    policy = mean_policy(state["actor_state_dict"])
    with gzip.open(native_path, "rt") as stream:
        native = [json.loads(line) for line in stream]
    if len(native) != len(rows):
        raise ValueError("Missing/extra native records or hidden reset")
    previous, history = None, np.zeros(4)
    for index, (row, record) in enumerate(zip(rows, native)):
        expected_kind = {"INITIAL": "RESET", "RETARGET": "RETARGET", "STEP": "STEP"}[row["kind"]]
        if record["kind"] != expected_kind or record["episode_id"] != 1 or record["step"] != row["step"]:
            raise ValueError("Native command/step/reset differs")
        actual = record["actual"]
        match_native_sample(row["sample"], actual, properties)
        if index == 0:
            verify_initial_sample(actual, case["initial"])
            if np.any(record["actions"]) or actual["time_s"] != 0 or np.any(actual["rpm"]):
                raise ValueError("Initial action/input state differs")
        else:
            if record["target_id"] != str(row["target_index"]) or not np.array_equal(
                record["target"], row["target"]
            ):
                raise ValueError("Native actual command differs")
            if expected_kind == "RETARGET":
                if actual != previous or not np.array_equal(record["actions"], history):
                    raise ValueError("Retarget changed native bodies/joints/time/held forces/history")
            else:
                obs = actor_observation(
                    observation_sample(previous, case["side_m"]), record["target"], history
                )
                if not np.allclose(record["actor_observation_before_step"], obs, rtol=0, atol=1e-10):
                    raise ValueError("Actual actor observation differs from measured input/history")
                expected_action = np.clip(policy(obs), -1, 1)
                if not np.allclose(record["actions"], expected_action, rtol=0, atol=1e-7):
                    raise ValueError("Applied action differs from exact frozen actor")
                if not np.allclose(
                    actual["rpm"],
                    (1 + 0.8 * expected_action.astype(float)) * 14468.429183500699,
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    raise ValueError("Current-action original motor mapping differs")
                history = np.asarray(record["actions"])
        if not np.array_equal(row["sample"]["actions"], record["actions"]):
            raise ValueError("Saved applied action history differs")
        previous = actual
    return dict(
        native_raw_sha256=sha256(native_path),
        actual_properties_sha256=sha256(properties_path),
        checkpoint_sha256=expected_hash,
        records=len(native),
        actor_outputs_reconstructed=True,
    )
