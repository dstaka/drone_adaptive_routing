"""Read-only modal diagnostics; no live controller or acceptance decisions.

Cross spectra in feedback are descriptive phase relationships. They cannot
identify a plant's causal transport delay without an independent excitation.
"""

import numpy as np


def modal_summary(times, signal):
    t, y = np.asarray(times, float), np.asarray(signal, float)
    result = dict(
        dominant_hz=None,
        amplitude_envelope_rms=[],
        envelope_log_decay_per_s=None,
        decay_identification="INSUFFICIENT_WINDOWS",
    )
    if len(t) < 200 or y.shape != t.shape or not np.isfinite(y).all():
        return result
    if not np.allclose(np.diff(t), 0.01, rtol=0, atol=1e-8):
        raise ValueError("Complete100Hz intervals required")
    centered = y - y.mean()
    if np.max(np.abs(centered)) < 1e-6:
        result["decay_identification"] = "UNEXCITED"
        return result
    f = np.fft.rfftfreq(len(y), 0.01)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(y)))) ** 2
    power[f < 0.1] = 0
    result["dominant_hz"] = float(f[np.argmax(power)])
    envelope = [
        (float(t[i : i + 100].mean()), float(np.sqrt(np.mean(y[i : i + 100] ** 2))))
        for i in range(0, len(y) - 99, 100)
    ]
    result["amplitude_envelope_rms"] = envelope
    usable = [(x, a) for x, a in envelope if a > 1e-6]
    if len(usable) >= 4:
        x, a = np.array(usable).T
        slope, intercept = np.polyfit(x, np.log(a), 1)
        residual = np.sum((np.log(a) - intercept - slope * x) ** 2)
        total = np.sum((np.log(a) - np.log(a).mean()) ** 2)
        result.update(
            envelope_log_decay_per_s=float(slope),
            envelope_log_fit_r2=float(1 - residual / total) if total > 1e-12 else None,
            decay_identification="DESCRIPTIVE_RMS_ENVELOPE_NOT_A_CAUSAL_DAMPING_FIT",
        )
    return result


def command_phase(command, motion, frequency_hz):
    """Welch phase near the measured mode, with explicit identifiability gates."""
    u, y = np.asarray(command, float), np.asarray(motion, float)
    result = dict(
        identified=False,
        reason="INSUFFICIENT_INDEPENDENT_EXCITATION",
        causal_transport_delay="UNIDENTIFIABLE_CLOSED_LOOP",
    )
    if (
        u.shape != y.shape
        or u.ndim != 1
        or len(u) < 400
        or not np.isfinite(u).all()
        or not np.isfinite(y).all()
        or frequency_hz is None
        or not 0.1 <= frequency_hz < 49
    ):
        return result
    if np.std(u) < 1e-9 or np.std(y) < 1e-6:
        return result
    length = 200
    windows = np.hanning(length)
    uf, yf = [], []
    for start in range(0, len(u) - length + 1, length // 2):
        a, b = u[start : start + length], y[start : start + length]
        uf.append(np.fft.rfft((a - a.mean()) * windows))
        yf.append(np.fft.rfft((b - b.mean()) * windows))
    uf, yf = np.array(uf), np.array(yf)
    pu, py = (np.mean(np.abs(x) ** 2, axis=0) for x in (uf, yf))
    cross = np.mean(uf.conj() * yf, axis=0)
    f = np.fft.rfftfreq(length, 0.01)
    index = max(1, int(np.argmin(np.abs(f - frequency_hz))))
    coherence = float(abs(cross[index]) ** 2 / max(pu[index] * py[index], 1e-300))
    input_fraction = float(pu[index] / max(pu[1:].sum(), 1e-300))
    motion_fraction = float(py[index] / max(py[1:].sum(), 1e-300))
    result.update(
        frequency_hz=float(f[index]),
        coherence=coherence,
        command_spectral_fraction=input_fraction,
        motion_spectral_fraction=motion_fraction,
        overlapping_segments=len(uf),
    )
    if coherence < 0.8 or input_fraction < 0.05 or motion_fraction < 0.05:
        return result
    phase = float(np.angle(cross[index]))
    result.update(
        identified=True,
        reason="DESCRIPTIVE_CLOSED_LOOP_CROSS_SPECTRUM",
        phase_rad=phase,
        phase_equivalent_lag_s=-phase / (2 * np.pi * f[index]),
        lag_ambiguity_s=float(1 / f[index]),
    )
    return result
