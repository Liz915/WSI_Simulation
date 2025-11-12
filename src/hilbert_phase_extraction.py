import numpy as np
from scipy.signal import hilbert

def extract_phase_hilbert(I):
    """基于Hilbert变换的相位提取"""
    analytic_signal = hilbert(I)
    phase = np.unwrap(np.angle(analytic_signal))
    return phase