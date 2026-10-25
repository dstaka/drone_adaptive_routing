import numpy as np

from adaptive_drone_patrol.payload_hybrid_diagnosis import modal_summary, command_phase


def test_frequency_and_closed_loop_phase_are_measured_without_claiming_causal_delay():
    t = np.arange(1200) * 0.01
    u = np.sin(2 * np.pi * 2 * t)
    y = np.sin(2 * np.pi * 2 * (t - 0.05))
    assert abs(modal_summary(t, y)["dominant_hz"] - 2) < 0.1
    result = command_phase(u, y, 2.0)
    assert result["identified"]
    assert abs(result["phase_equivalent_lag_s"] - 0.05) < 0.005
    assert result["causal_transport_delay"] == "UNIDENTIFIABLE_CLOSED_LOOP"


def test_short_or_unexcited_traces_do_not_invent_phase_or_decay():
    assert not command_phase(np.zeros(1200), np.ones(1200), 2.0)["identified"]
    t = np.arange(300) * 0.01
    assert modal_summary(t, np.sin(2 * np.pi * 2 * t))["envelope_log_decay_per_s"] is None
    assert not command_phase(np.ones(300), np.ones(300), 2.0)["identified"]
