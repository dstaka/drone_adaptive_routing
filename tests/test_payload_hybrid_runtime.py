import numpy as np
import pytest

from adaptive_drone_patrol.payload_hybrid_runtime import SmoothReference, diagnostic_summary


def test_reference_has_real_target_clock_independent_bounded_motion():
    r = SmoothReference([0,0,1], [5,0,1])
    assert r.sample(0)[0] == pytest.approx([0,0,1])
    assert r.sample(r.duration)[0] == pytest.approx([5,0,1])
    times = np.linspace(0,r.duration,1001)
    samples = [r.sample(t) for t in times]
    assert max(np.linalg.norm(v) for p,v in samples) <= 1.2+1e-8
    assert r.sample(100)[1] == pytest.approx([0,0,0])


def test_frequency_diagnosis_does_not_fix_mode_at_old_observation():
    t = np.arange(3000)*.01
    y = np.column_stack([np.sin(2*np.pi*3*t)*np.exp(-.2*t), np.zeros_like(t), np.zeros_like(t)])
    result = diagnostic_summary(t, y, np.zeros((3000,4)))
    assert result['dominant_hz'] == pytest.approx(3., abs=.04)
    assert result['envelope_log_decay_per_s'] < -.15
    assert diagnostic_summary(t,np.zeros_like(y),np.zeros((3000,4)))['dominant_hz'] is None
