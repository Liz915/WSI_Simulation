import numpy as np

def add_noise(I, z, vib_amp=20e-9, vib_freq=10, snr_db=30, lambda_c=600e-9):
    """
    加入机械振动 + 探测器噪声
    vib_amp: 振动幅度（m）
    vib_freq: 振动频率（Hz）
    snr_db: 信噪比（dB）
    """
    # 机械振动引起的相位扰动
    phi_vib = (4 * np.pi / lambda_c) * vib_amp * np.sin(2 * np.pi * vib_freq * z / z.max())
    
    # 直接在相位域施加扰动
    I_vib = I * np.cos(phi_vib)
    
    # 计算噪声功率并叠加高斯白噪声
    noise_power = np.mean(I_vib**2) / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(I))
    I_noisy = I_vib + noise

    return I_noisy