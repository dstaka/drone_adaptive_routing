"""Independent full powered checking fails closed on absent or stale evidence."""

import pytest


def test_missing_powered_raw_cannot_qualify(tmp_path):
    from adaptive_drone_patrol.payload_solver_qualification import verify_powered_probe, make_manifest

    m = make_manifest()
    case = next(c for c in m["cases"] if c["kind"] == "powered")
    with pytest.raises(ValueError):
        verify_powered_probe(tmp_path, m, case)


def test_incomplete_refinement_pairs_are_not_successes(tmp_path):
    from adaptive_drone_patrol.payload_solver_qualification import verify_refinement_pairs, make_manifest

    with pytest.raises(ValueError):
        verify_refinement_pairs(tmp_path, make_manifest())
