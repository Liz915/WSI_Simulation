# main.py
"""
WSI 3D表面重建主程序 - 统一架构版本
用于批处理测试和CI验证
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# 从统一的核心模块导入
from src.signal_generator import create_simulated_surface, simulate_wsi_stack_3d
from src.noise_model import add_noise_3d
from src.processing import process_cps_subpixel, process_fft_phase
from src.phase_unwrap import unwrap_surface_2d
from src.visualization import plot_surface, plot_interferogram

def main_3d_simulation():
    print("🚀 开始WSI 3D表面重建 (统一架构版本)...")
    print("=" * 60)
    
    # --- 1. 定义仿真参数 ---
    Z_SCAN = np.linspace(-2e-6, 2e-6, 256)  # Z轴扫描 (256步)
    SURFACE_SHAPE = (128, 128)              # 表面尺寸 (y, x)
    STEP_HEIGHT_NM = 40.0                   # 40nm台阶
    LAMBDA_C = 600e-9                       # 中心波长
    LC = 0.8e-6                             # 相干长度
    
    print(f"📐 仿真参数:")
    print(f"  表面尺寸: {SURFACE_SHAPE}")
    print(f"  台阶高度: {STEP_HEIGHT_NM} nm")
    print(f"  Z轴扫描: {len(Z_SCAN)}点, 范围: ±2μm")
    print(f"  中心波长: {LAMBDA_C*1e9:.0f} nm")
    
    # --- 2. 生成模拟数据 ---
    print(f"\n📊 STEP 1: 生成{SURFACE_SHAPE}模拟表面...")
    ground_truth_surface = create_simulated_surface(
        shape=SURFACE_SHAPE, 
        step_height_nm=STEP_HEIGHT_NM
    )
    
    print(f"📊 STEP 2: 生成3D WSI信号栈...")
    start_time = time.time()
    ideal_stack = simulate_wsi_stack_3d(
        ground_truth_surface, Z_SCAN, 
        lambda_c=LAMBDA_C, Lc=LC
    )
    print(f"  ✅ 理想信号栈生成完毕, 耗时: {time.time() - start_time:.2f}秒")
    
    # --- 3. 注入真实噪声 ---
    print(f"\n🌪️ STEP 3: 注入生产环境噪声 (10nm振动, 30dB SNR)...")
    noisy_stack, vib_displacement = add_noise_3d(
        ideal_stack, Z_SCAN, ground_truth_surface,
        vib_amp_nm=10.0,    # 真实生产环境振动
        snr_db=30,          # 真实信噪比
        lambda_c=LAMBDA_C
    )
    print("  ✅ 噪声注入完成")
    
    # 绘制中心像素干涉图
    center_y, center_x = SURFACE_SHAPE[0] // 2, SURFACE_SHAPE[1] // 2
    plot_interferogram(
        Z_SCAN,
        ideal_stack[:, center_y, center_x],
        noisy_stack[:, center_y, center_x],
        fname="main_interferogram_center_pixel.png"
    )
    
    # --- 4. 算法一：CPS重建 ---
    print(f"\n🔧 STEP 4A: 使用CPS算法处理...")
    start_time = time.time()
    height_map_cps, coherence_cps = process_cps_subpixel(
        noisy_stack, Z_SCAN, smooth_sigma=8.0
    )
    cps_time = time.time() - start_time
    print(f"  ✅ CPS算法完成, 耗时: {cps_time:.2f}秒")
    
    # CPS高度转换和调平
    height_cps_nm = height_map_cps * 1e9  # 转换为纳米
    background_mask = (ground_truth_surface == 0)
    height_cps_nm -= np.mean(height_cps_nm[background_mask])
    
    # --- 5. 算法二：FFT相位重建 ---
    print(f"\n🔧 STEP 4B: 使用FFT相位算法处理...")
    start_time = time.time()
    wrapped_phase, coherence_fft = process_fft_phase(
        noisy_stack, Z_SCAN, smooth_sigma=10.0
    )
    
    print(f"  🔧 STEP 5: 2D相位解包裹...")
    unwrapped_phase = unwrap_surface_2d(wrapped_phase)
    
    # 相位到高度转换
    height_fft_nm = unwrapped_phase * (LAMBDA_C * 1e9 / (4 * np.pi))
    height_fft_nm -= np.mean(height_fft_nm[background_mask])
    
    fft_time = time.time() - start_time
    print(f"  ✅ FFT相位算法完成, 总耗时: {fft_time:.2f}秒")
    
    # --- 6. 结果可视化 ---
    print(f"\n📈 STEP 6: 生成结果图...")
    
    # 绘制CPS结果
    plot_surface(
        np.arange(SURFACE_SHAPE[1]),
        np.arange(SURFACE_SHAPE[0]),
        height_cps_nm,
        fname="reconstructed_surface_CPS.png"
    )
    
    # 绘制FFT结果  
    plot_surface(
        np.arange(SURFACE_SHAPE[1]),
        np.arange(SURFACE_SHAPE[0]),
        height_fft_nm,
        fname="reconstructed_surface_FFT.png"
    )
    
    # --- 7. 性能分析 ---
    print(f"\n📊 STEP 7: 算法性能对比...")
    
    def calculate_metrics(height_map_nm, ground_truth_nm, algorithm_name):
        """计算算法性能指标"""
        background_mask = (ground_truth_nm == 0)
        step_mask = (ground_truth_nm > 0)
        
        step_height = np.mean(height_map_nm[step_mask]) - np.mean(height_map_nm[background_mask])
        background_std = np.std(height_map_nm[background_mask])
        rmse = np.sqrt(np.mean((height_map_nm - ground_truth_nm)**2))
        
        print(f"  {algorithm_name}:")
        print(f"    重建台阶高度: {step_height:.2f} nm")
        print(f"    背景噪声: {background_std:.2f} nm")
        print(f"    全局RMSE: {rmse:.2f} nm")
        
        return step_height, background_std, rmse
    
    ground_truth_nm = ground_truth_surface * 1e9
    
    print("  " + "="*40)
    cps_step, cps_noise, cps_rmse = calculate_metrics(height_cps_nm, ground_truth_nm, "CPS算法")
    fft_step, fft_noise, fft_rmse = calculate_metrics(height_fft_nm, ground_truth_nm, "FFT相位算法")
    print("  " + "="*40)
    
    print(f"\n⏱️ 计算速度对比:")
    print(f"  CPS算法: {cps_time:.2f}秒")
    print(f"  FFT相位算法: {fft_time:.2f}秒")
    print(f"  速度比: {fft_time/cps_time:.1f}x")
    
    print("\n🎉 3D仿真流程全部完成！")
    print("📁 请检查 'data/example_output' 文件夹查看结果")

if __name__ == "__main__":
    main_3d_simulation()