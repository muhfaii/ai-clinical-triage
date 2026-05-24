import numpy as np
from scipy.signal import butter, filtfilt


def butterworth_lowpass(signal: np.ndarray, cutoff: float = 6.0, fs: float = 30.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter. cutoff in Hz, fs in Hz."""
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # filtfilt needs signal length > padlen; fall back to raw if too short
    if len(signal) < 3 * max(len(a), len(b)):
        return signal.copy()
    return filtfilt(b, a, signal)
