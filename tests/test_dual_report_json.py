"""Administrative formatting only; these tests grant no physical admission."""

import importlib.util
import json
from pathlib import Path
import pytest


def adapter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "dual_report_json.py"
    spec = importlib.util.spec_from_file_location("dual_report_json", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.json_report


def test_nested_supplement_checks_have_stable_lossless_json():
    convert = adapter()
    source = dict(
        passed=False,
        errors=["gate remains unmet"],
        evidence=[dict(checks={"streaming", "address_actuation"}, details=None), dict(checks=set())],
    )
    result = convert(source)
    assert result == dict(
        passed=False,
        errors=["gate remains unmet"],
        evidence=[dict(checks=["address_actuation", "streaming"], details=None), dict(checks=[])],
    )
    assert isinstance(source["evidence"][0]["checks"], set)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("invalid", [{1, "check"}, object(), Path("unbound"), {1: "key"}])
def test_unexpected_values_are_not_silently_stringified(invalid):
    with pytest.raises(TypeError):
        adapter()(invalid)
