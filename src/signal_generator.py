import numpy as np

def simulate_wsi_signal(z, lambda_c=600e-9, Lc=1e-6, h0=0, Idc=1, A0=0.6):
    """生成理想白光干涉信号"""
    envelope = A0 * np.exp(-((z - h0)**2) / (Lc**2))
    phase = 4 * np.pi / lambda_c * (z - h0)
    I = Idc + envelope * np.cos(phase)
    return I