import numpy as np


def trunk_angle_deg(shoulder_xy: np.ndarray, hip_xy: np.ndarray) -> np.ndarray:
    """
    Trunk flexion angle from vertical (degrees).

    Works in image coordinates (y↓). The trunk vector points from hip to shoulder.
    Angle from vertical = arctan2(|Δx|, |Δy|), so upright posture → ~0°.
    """
    vec = shoulder_xy - hip_xy  # (N, 2)
    return np.degrees(np.arctan2(np.abs(vec[:, 0]), np.abs(vec[:, 1])))


def angular_velocity(angles: np.ndarray, fs: float) -> np.ndarray:
    return np.gradient(angles, 1.0 / fs)
