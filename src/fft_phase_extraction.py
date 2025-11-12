import numpy as np
from scipy.fft import fft, ifft, fftfreq

def extract_phase_fft(I, z):
    """基于FFT的相位提取"""
    N = len(I)
    F = fft(I)
    freq = fftfreq(N, d=(z[1]-z[0]))
    # 选取正频段
    mask = (freq > 0)
    F_filtered = np.zeros_like(F)
    F_filtered[mask] = F[mask]
    I_filtered = ifft(F_filtered)
    phase = np.unwrap(np.angle(I_filtered))
    return phase