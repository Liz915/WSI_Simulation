# src/noise_model.py
import numpy as np

def add_noise_3d(stack, z_scan, surface, lambda_c=600e-9, vib_amp_nm=10.0, vib_freq_hz=50.0, snr_db=30.0, add_poisson=False, max_photons=1e4):
    """
    完整的高噪声模型（来自你 main.py 的版本）
    """
    n_z, n_y, n_x = stack.shape
    
    # 1. 重新构造带振动的干涉信号
    z_3d = z_scan.reshape(n_z, 1, 1)
    surface_3d = surface.reshape(1, n_y, n_x)
    
    # 振动位移 (单位: 米)
    t = np.linspace(0, 1, n_z) # 假设扫描时间为1个单位
    vib_displacement = vib_amp_nm * 1e-9 * np.sin(2 * np.pi * vib_freq_hz * t)
    vib_displacement_3d = vib_displacement.reshape(n_z, 1, 1)
    
    # 带振动的光程差
    opd = (z_3d + vib_displacement_3d) - surface_3d
    
    # 重新生成信号
    # (我们应该使用原始信号，而不是重新生成它。让我们修改一下)
    # (更正的逻辑：我们应该在原始 'stack' 上添加噪声)
    
    # 1. 振动引起的相位噪声
    k0 = 4 * np.pi / lambda_c # 4*pi 因为是往返
    t = np.linspace(0, 1, n_z)
    vib_displacement = vib_amp_nm * 1e-9 * np.sin(2 * np.pi * vib_freq_hz * t)
    vib_phase_noise = np.cos(k0 * vib_displacement).reshape(n_z, 1, 1)
    
    # 将振动噪声作为乘性噪声
    stack_vib = stack * vib_phase_noise
    
    # 2. 添加加性高斯白噪声 (AWGN)
    signal_power = np.mean(stack_vib**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise_std = np.sqrt(noise_power)
    awgn = np.random.normal(0, noise_std, size=stack.shape)
    
    noisy_stack = stack_vib + awgn
    
    # 3. 光子噪声 (可选)
    if add_poisson:
        signal_mean = np.mean(noisy_stack)
        scaling_factor = max_photons / signal_mean
        photon_signal = noisy_stack * scaling_factor
        photon_noisy = np.random.poisson(np.clip(photon_signal, 0, None)) # 确保泊松输入为非负
        noisy_stack = photon_noisy / scaling_factor
    
    # 4. 模拟饱和 (保持在合理范围)
    noisy_stack = np.clip(noisy_stack, 0, 2) # 假设最大强度为2
    
    return noisy_stack, vib_displacement_3d