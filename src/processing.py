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
    算法二：基于FFT频域载波相位的WLPSI重建 (Takeda 修正版)
    （适用于高精度离线计量）
    
    参数:
        stack: 3D干涉信号栈 (n_z, n_y, n_x)
        z_scan: Z轴扫描位置数组 (n_z,)
        smooth_sigma: (此方法中未使用，为保持接口一致性保留)
        band_frac: (此方法中未使用，为保持接口一致性保留)
        
    返回:
        wrapped_phase_map: 包裹相位图 (n_y, n_x) [-π, π]
        coherence_map: 相干度图 (n_y, n_x)
    """
    print(f"🔧 开始FFT相位算法处理 (Takeda 修正版): 栈尺寸{stack.shape}")
    n_z, n_y, n_x = stack.shape
    
    if n_z < 3:
        raise ValueError("需要至少3个Z轴采样点")

    # 1. 计算Z轴步长 (dz)
    dz = float(z_scan[1] - z_scan[0])
    
    # 2. 沿Z轴进行FFT
    stack_fft = fft(stack, axis=0)
    freqs = fftfreq(n_z, d=dz)
    
    # 3. 找到正频率的载波频率 (关键步骤)
    # 我们只关心正频率部分 (k > 0)，因为负频率是共轭的
    positive_freq_mask = (freqs > 0)
    
    # 如果没有正频率 (例如采样点太少)，则出错
    if not np.any(positive_freq_mask):
        raise ValueError("无法找到正载波频率，请检查Z轴采样")
        
    # 计算正频率部分的平均频谱
    mean_spectrum = np.mean(np.abs(stack_fft[positive_freq_mask, :, :]), axis=(1, 2))
    
    # 找到正频率中的峰值索引（相对于掩码）
    center_idx_relative = np.argmax(mean_spectrum)
    
    # 将其映射回原始FFT数组的绝对索引
    positive_indices = np.where(positive_freq_mask)[0]
    center_idx_absolute = positive_indices[center_idx_relative]
    
    print(f"  ...检测到载波频率索引: {center_idx_absolute} (对应频率: {freqs[center_idx_absolute]:.2f})")
    
    # 4. 提取该频率下的相位和相干度 (核心)
    
    # 包裹相位图 = 该载波频率分量的相位角
    wrapped_phase_map = np.angle(stack_fft[center_idx_absolute, :, :])
    
    # 相干度图 = 该载波频率分量的幅度
    coherence_map = np.abs(stack_fft[center_idx_absolute, :, :])

    print("✅ FFT相位算法 (Takeda 修正版) 处理完成")
    return wrapped_phase_map, coherence_map