import numpy as np
import pytest

from sit2stand_ml.pipeline.filter import butterworth_lowpass
from sit2stand_ml.pipeline.kinematics import angular_velocity, trunk_angle_deg
from sit2stand_ml.pipeline.pipeline import analyze
from sit2stand_ml.pipeline.segmenter import detect_reps


def _synthetic_sts(n_reps: int = 3, fps: float = 30.0) -> list[dict]:
    """
    Synthetic sit-to-stand frames. Hip oscillates between y=400 (seated) and y=200
    (standing) in image coordinates (y↓). Shoulder is 150 px above hip with a slight
    forward lean during the movement.
    """
    T = int(fps * 2 * n_reps)
    t = np.linspace(0, 2 * np.pi * n_reps, T)

    hip_y = 300 + 100 * np.cos(t)
    hip_x = np.full(T, 320.0)
    shoulder_y = hip_y - 150
    shoulder_x = hip_x + 15 * np.sin(t)  # forward lean during transition
    knee_y = hip_y + 100
    ankle_y = np.full(T, 600.0)

    return [
        {
            "hip": [hip_x[i], hip_y[i]],
            "knee": [hip_x[i], knee_y[i]],
            "ankle": [hip_x[i], ankle_y[i]],
            "shoulder": [shoulder_x[i], shoulder_y[i]],
        }
        for i in range(T)
    ]


# --- filter ---

def test_butterworth_passes_dc():
    signal = np.ones(100)
    out = butterworth_lowpass(signal, cutoff=6.0, fs=30.0)
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_butterworth_attenuates_high_freq():
    fs = 30.0
    t = np.arange(200) / fs
    low = np.sin(2 * np.pi * 1.0 * t)   # 1 Hz — passes
    high = np.sin(2 * np.pi * 12.0 * t)  # 12 Hz — attenuated
    low_out = butterworth_lowpass(low + high, cutoff=6.0, fs=fs)
    assert np.std(low_out - low) < 0.1


def test_butterworth_short_signal_fallback():
    signal = np.array([1.0, 2.0, 3.0])
    out = butterworth_lowpass(signal, cutoff=6.0, fs=30.0)
    np.testing.assert_array_equal(out, signal)


# --- kinematics ---

def test_trunk_angle_upright():
    shoulder = np.array([[100.0, 50.0]])
    hip = np.array([[100.0, 200.0]])
    angles = trunk_angle_deg(shoulder, hip)
    assert angles[0] == pytest.approx(0.0, abs=1e-6)


def test_trunk_angle_leaning():
    shoulder = np.array([[200.0, 100.0]])
    hip = np.array([[100.0, 200.0]])
    angles = trunk_angle_deg(shoulder, hip)
    assert angles[0] == pytest.approx(45.0, abs=1e-6)


def test_angular_velocity_constant_angle():
    angles = np.full(60, 30.0)
    vel = angular_velocity(angles, fs=30.0)
    np.testing.assert_allclose(vel, 0.0, atol=1e-6)


# --- segmenter ---

def test_detect_reps_count():
    frames = _synthetic_sts(n_reps=3, fps=30.0)
    hip_y = np.array([f["hip"][1] for f in frames])
    reps = detect_reps(hip_y, fs=30.0)
    assert len(reps) == 3


def test_detect_reps_flat_signal():
    hip_y = np.full(90, 300.0)
    reps = detect_reps(hip_y, fs=30.0)
    assert reps == []


# --- full pipeline ---

def test_analyze_rep_count():
    frames = _synthetic_sts(n_reps=3, fps=30.0)
    result = analyze(frames, frame_rate=30.0)
    assert result["rep_count"] == 3


def test_analyze_movement_time_reasonable():
    frames = _synthetic_sts(n_reps=1, fps=30.0)
    result = analyze(frames, frame_rate=30.0)
    # Synthetic rep is ~2 s
    assert 1.0 <= result["mean_movement_time_s"] <= 3.0


def test_analyze_trunk_flexion_nonzero():
    frames = _synthetic_sts(n_reps=1, fps=30.0)
    result = analyze(frames, frame_rate=30.0)
    assert result["mean_trunk_flexion_deg"] > 0


def test_analyze_requires_minimum_frames():
    with pytest.raises(ValueError, match="10 frames"):
        analyze([{"hip": [0, 0], "knee": [0, 0], "ankle": [0, 0], "shoulder": [0, 0]}] * 5)


def test_analyze_reps_structure():
    frames = _synthetic_sts(n_reps=2, fps=30.0)
    result = analyze(frames, frame_rate=30.0)
    for rep in result["reps"]:
        assert "duration_s" in rep
        assert "peak_trunk_flexion_deg" in rep
        assert "peak_angular_velocity_deg_s" in rep
