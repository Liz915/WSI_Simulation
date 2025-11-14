# src/fft_phase_extraction.py
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d

def _parabolic_subpixel(vm1, v0, vp1):
    """
    基于二次曲线拟合的亚像素峰值偏移量。
    （来自你的原始代码）
    """
    denom = (vm1 - 2 * v0 + vp1)
    if np.abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (vm1 - vp1) / denom

def process_stack_fft(stack, z_scan, smooth_sigma=10.0, band_frac=0.15):
    """
    高级算法：使用FFT构造解析信号并提取包裹相位。
    注意：我们将默认的 'smooth_sigma' 从 1.0 提高到 10.0。
    这是对抗高噪声的关键！
    """
    n_z, n_y, n_x = stack.shape
    if n_z < 3:
        raise ValueError("Need at least 3 z samples.")

    # --- 1. dz 修复（标量） ---
    dz = float(z_scan[1] - z_scan[0])

    # --- 2. FFT 沿 z 轴 ---
    stack_fft = fft(stack, axis=0)
    freqs = fftfreq(n_z, d=dz)

    # --- 3. 自适应带通设计 ---
    mean_spectrum = np.mean(np.abs(stack_fft), axis=(1,2))
    center_idx = np.argmax(mean_spectrum)
    half_bw = max(2, int(n_z * band_frac / 2))
    band = np.zeros_like(freqs, dtype=float)
    idxs = np.arange(n_z)
    sigma = max(1.0, half_bw / 2.0)
    band = np.exp(-0.5 * ((idxs - center_idx) / sigma)**2)
    analytic_mask = np.zeros_like(freqs, dtype=float)
    analytic_mask[freqs > 0] = 2.0
    analytic_mask[np.isclose(freqs, 0.0)] = 1.0
    filter_1d = band * analytic_mask
    filter_3d = filter_1d.reshape(n_z, 1, 1)

    # --- 4. 应用滤波器并 IFFT 回时域（解析信号） ---
    stack_fft_filtered = stack_fft * filter_3d
    analytic_stack = ifft(stack_fft_filtered, axis=0)  # 复数解析信号

    # --- 5. 包络与相位 ---
    envelope = np.abs(analytic_stack)
    phase_stack = np.angle(analytic_stack)

    # --- 6. 平滑包络（关键步骤！） ---
    # 使用传入的 'smooth_sigma' (例如 10.0)
    envelope_smooth = gaussian_filter1d(envelope, sigma=smooth_sigma, axis=0, mode='nearest')

    # --- 7. 找峰值索引（整数） ---
    peak_idx = np.argmax(envelope_smooth, axis=0)  # shape (n_y, n_x)

    # --- 8. 亚像素偏移量（parabolic fit）和复数插值获取相位 ---
    y_idx, x_idx = np.indices((n_y, n_x))
    wrapped_phase_map = np.zeros((n_y, n_x), dtype=float)
    coherence_map = np.zeros((n_y, n_x), dtype=float)
    
    z_indices = np.arange(n_z) # 为插值做准备

    for yi in range(n_y):
        for xi in range(n_x):
            i = int(peak_idx[yi, xi])
            if i <= 0: i = 1
            if i >= n_z - 1: i = n_z - 2

            vm1 = envelope_smooth[i - 1, yi, xi]
            v0  = envelope_smooth[i    , yi, xi]
            vp1 = envelope_smooth[i + 1, yi, xi]

            shift = _parabolic_subpixel(vm1, v0, vp1)
            float_idx = i + shift

            real_seq = analytic_stack[:, yi, xi].real
            imag_seq = analytic_stack[:, yi, xi].imag
            real_val = np.interp(float_idx, z_indices, real_seq)
            imag_val = np.interp(float_idx, z_indices, imag_seq)
            complex_val = real_val + 1j * imag_val

            wrapped_phase_map[yi, xi] = np.angle(complex_val)
            env_val = np.interp(float_idx, z_indices, envelope[:, yi, xi])
            coherence_map[yi, xi] = env_val

    return wrapped_phase_map, coherence_map