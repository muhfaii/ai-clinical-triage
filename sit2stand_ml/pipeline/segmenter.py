import numpy as np
from scipy.signal import find_peaks


def detect_reps(hip_y: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """
    Detect sit-to-stand repetitions from the hip vertical trajectory.

    In image coordinates y↓, standing = hip moves up = hip_y decreases.
    We invert to a 'height' signal (standing = peaks) then find peak/valley pairs.

    Returns list of (start_frame, end_frame) for each detected rep.
    """
    height = -hip_y
    ptp = float(np.ptp(height))
    if ptp < 1e-6:
        return []

    prominence = 0.2 * ptp
    min_distance = max(1, int(fs * 0.5))  # reps at least 0.5 s apart

    peaks, _ = find_peaks(height, prominence=prominence, distance=min_distance)
    valleys, _ = find_peaks(-height, prominence=prominence, distance=min_distance)

    if len(peaks) == 0:
        return []

    reps = []
    for peak_idx in peaks:
        before = valleys[valleys < peak_idx]
        after = valleys[valleys > peak_idx]
        start = int(before[-1]) if len(before) > 0 else 0
        end = int(after[0]) if len(after) > 0 else len(hip_y) - 1
        reps.append((start, end))

    return reps
