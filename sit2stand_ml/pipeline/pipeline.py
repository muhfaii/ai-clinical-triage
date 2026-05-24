import numpy as np

from .filter import butterworth_lowpass
from .kinematics import angular_velocity, trunk_angle_deg
from .segmenter import detect_reps


def analyze(frames: list[dict], frame_rate: float = 30.0) -> dict:
    """
    Compute sit-to-stand kinematic metrics from a sequence of pose keyframes.

    Each frame must contain keys: hip, knee, ankle, shoulder — each a [x, y] pair
    in any consistent coordinate unit (pixels, normalised, etc.).

    Returns a dict with per-rep and aggregate metrics.
    """
    if len(frames) < 10:
        raise ValueError("At least 10 frames required for analysis")

    hip = np.array([f["hip"] for f in frames], dtype=float)        # (N, 2)
    shoulder = np.array([f["shoulder"] for f in frames], dtype=float)  # (N, 2)

    # Filter hip_y for segmentation
    hip_y_filt = butterworth_lowpass(hip[:, 1], cutoff=6.0, fs=frame_rate)

    # Compute trunk angle and filter
    raw_angles = trunk_angle_deg(shoulder, hip)
    angles_filt = butterworth_lowpass(raw_angles, cutoff=6.0, fs=frame_rate)

    reps = detect_reps(hip_y_filt, frame_rate)

    if reps:
        rep_metrics = [_rep_metrics(angles_filt, start, end, frame_rate) for start, end in reps]
    else:
        # Treat the entire sequence as a single movement
        rep_metrics = [_rep_metrics(angles_filt, 0, len(frames) - 1, frame_rate)]

    means = {
        "rep_count": len(rep_metrics),
        "mean_movement_time_s": round(float(np.mean([r["duration_s"] for r in rep_metrics])), 3),
        "mean_trunk_flexion_deg": round(float(np.mean([r["peak_trunk_flexion_deg"] for r in rep_metrics])), 2),
        "mean_angular_velocity_deg_s": round(float(np.mean([r["peak_angular_velocity_deg_s"] for r in rep_metrics])), 2),
        "reps": rep_metrics,
    }
    return means


def _rep_metrics(angles: np.ndarray, start: int, end: int, fs: float) -> dict:
    segment = angles[start : end + 1]
    if len(segment) == 0:
        segment = angles

    vel = angular_velocity(segment, fs)
    return {
        "duration_s": round((end - start) / fs, 3),
        "peak_trunk_flexion_deg": round(float(np.max(segment)), 2),
        "peak_angular_velocity_deg_s": round(float(np.max(np.abs(vel))), 2),
    }
