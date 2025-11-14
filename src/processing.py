# src/processing.py
"""
白光干涉信号处理核心模块
包含两种工业标准算法：CPS（相干峰值搜寻）和 FFT相位提取
专为高噪声生产环境优化
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

def _parabolic_subpixel(vm1, v0, vp1):
    """
    基于二次曲线拟合的亚像素峰值偏移量计算
    
    参数:
        vm1, v0, vp1: 峰值附近三个点的包络值
        
    返回:
        shift: 亚像素偏移量 (-0.5 到 0.5)
    """
    denom = (vm1 - 2 * v0 + vp1)
    if np.abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (vm1 - vp1) / denom

def process_cps_subpixel(stack, z_scan, smooth_sigma=8.0):
    """
    算法一：基于时域Hilbert变换和亚像素包络峰值的CPS重建
    （适用于快速在线测量）
    
    参数:
        stack: 3D干涉信号栈 (n_z, n_y, n_x)
        z_scan: Z轴扫描位置数组 (n_z,)
        smooth_sigma: 包络平滑强度，建议8.0-12.0
        
    返回:
        height_map: 重建高度图 (n_y, n_x) [米]
        coherence_map: 相干度图 (n_y, n_x)
    """
    n_z, n_y, n_x = stack.shape
    height_map = np.zeros((n_y, n_x))
    coherence_map = np.zeros((n_y, n_x))
    
    print(f"🔧 开始CPS算法处理: 栈尺寸{stack.shape}, 平滑sigma={smooth_sigma}")
    
    # 1. 沿Z轴计算解析信号和包络
    analytic_stack = hilbert(stack, axis=0)
    envelope = np.abs(analytic_stack)
    
    # 2. 强力平滑包络 - 这是抗噪声的关键！
    envelope_smooth = gaussian_filter1d(envelope, sigma=smooth_sigma, axis=0, mode='nearest')
    
    # 3. 寻找整数峰值位置
    peak_idx_int = np.argmax(envelope_smooth, axis=0)
    
    # 4. 亚像素插值 - 消除量化误差的关键步骤
    z_indices = np.arange(n_z)
    
    for yi in range(n_y):
        for xi in range(n_x):
            i = peak_idx_int[yi, xi]
            
            # 边界保护：如果峰值在边界，直接使用整数位置
            if i <= 0 or i >= n_z - 1:
                height_map[yi, xi] = z_scan[i]
                coherence_map[yi, xi] = envelope_smooth[i, yi, xi]
                continue
            
            # 亚像素拟合：获取峰值附近三个点
            vm1 = envelope_smooth[i - 1, yi, xi]  # 峰值前一个点
            v0 = envelope_smooth[i, yi, xi]       # 峰值点  
            vp1 = envelope_smooth[i + 1, yi, xi]  # 峰值后一个点
            
            # 计算亚像素偏移量
            shift = _parabolic_subpixel(vm1, v0, vp1)
            float_idx = i + shift  # 浮点数索引
            
            # 线性插值得到精确高度
            height_map[yi, xi] = np.interp(float_idx, z_indices, z_scan)
            coherence_map[yi, xi] = np.interp(float_idx, z_indices, envelope[:, yi, xi])
    
    print("✅ CPS算法处理完成")
    return height_map, coherence_map

def process_fft_phase(stack, z_scan, smooth_sigma=10.0, band_frac=0.15):
    """
    算法二：基于FFT频域滤波和亚像素相位的WLPSI重建
    （适用于高精度离线计量）
    
    参数:
        stack: 3D干涉信号栈 (n_z, n_y, n_x)
        z_scan: Z轴扫描位置数组 (n_z,)
        smooth_sigma: 包络平滑强度，建议10.0-15.0
        band_frac: 频域滤波带宽比例
        
    返回:
        wrapped_phase_map: 包裹相位图 (n_y, n_x) [-π, π]
        coherence_map: 相干度图 (n_y, n_x)
    """
    n_z, n_y, n_x = stack.shape
    
    if n_z < 3:
        raise ValueError("需要至少3个Z轴采样点")
    
    print(f"🔧 开始FFT相位算法处理: 栈尺寸{stack.shape}, 平滑sigma={smooth_sigma}")
    
    # 1. 计算Z轴步长
    dz = float(z_scan[1] - z_scan[0])  # 修正：计算实际步长
    
    # 2. 沿Z轴进行FFT
    stack_fft = fft(stack, axis=0)
    freqs = fftfreq(n_z, d=dz)
    
    # 3. 自适应带通滤波器设计
    mean_spectrum = np.mean(np.abs(stack_fft), axis=(1, 2))
    center_idx = np.argmax(mean_spectrum)
    
    half_bw = max(2, int(n_z * band_frac / 2))
    sigma = max(1.0, half_bw / 2.0)
    
    idxs = np.arange(n_z)
    band = np.exp(-0.5 * ((idxs - center_idx) / sigma)**2)
    
    # 构造解析信号滤波器（抑制负频率）
    analytic_mask = np.zeros_like(freqs, dtype=float)
    analytic_mask[freqs > 0] = 2.0
    analytic_mask[np.isclose(freqs, 0.0)] = 1.0
    
    filter_1d = band * analytic_mask
    filter_3d = filter_1d.reshape(n_z, 1, 1)
    
    # 4. 应用滤波器并逆变换
    stack_fft_filtered = stack_fft * filter_3d  # 修正：定义stack_fft_filtered
    analytic_stack = ifft(stack_fft_filtered, axis=0)
    
    # 5. 提取包络和相位
    envelope = np.abs(analytic_stack)
    phase_stack = np.angle(analytic_stack)
    
    # 6. 强力平滑包络 - 关键改进！
    envelope_smooth = gaussian_filter1d(envelope, sigma=smooth_sigma, axis=0, mode='nearest')
    
    # 7. 寻找包络峰值
    peak_idx = np.argmax(envelope_smooth, axis=0)
    
    # 8. 亚像素相位插值
    wrapped_phase_map = np.zeros((n_y, n_x), dtype=float)  # 修正：定义返回变量
    coherence_map = np.zeros((n_y, n_x), dtype=float)      # 修正：定义返回变量
    
    z_indices = np.arange(n_z)  # 修正：定义z_indices
    
    for yi in range(n_y):
        for xi in range(n_x):
            i = int(peak_idx[yi, xi])
            
            # 边界保护
            if i <= 0:
                i = 1
            if i >= n_z - 1:
                i = n_z - 2
            
            # 亚像素拟合
            vm1 = envelope_smooth[i - 1, yi, xi]
            v0 = envelope_smooth[i, yi, xi]
            vp1 = envelope_smooth[i + 1, yi, xi]
            shift = _parabolic_subpixel(vm1, v0, vp1)
            float_idx = i + shift
            
            # 复数插值获取精确相位
            real_seq = analytic_stack[:, yi, xi].real
            imag_seq = analytic_stack[:, yi, xi].imag
            
            real_val = np.interp(float_idx, z_indices, real_seq)
            imag_val = np.interp(float_idx, z_indices, imag_seq)
            complex_val = real_val + 1j * imag_val
            
            wrapped_phase_map[yi, xi] = np.angle(complex_val)
            coherence_map[yi, xi] = np.interp(float_idx, z_indices, envelope[:, yi, xi])
    
    print("✅ FFT相位算法处理完成")
    return wrapped_phase_map, coherence_map  # 修正：返回已定义的变量