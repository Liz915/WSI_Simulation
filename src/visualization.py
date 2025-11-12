# src/visualization.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 从 utils 导入统一的输出目录
# 动态导入 utils，兼容各种运行环境
def _import_utils():
    """动态导入 utils 模块"""
    try:
        # 尝试直接导入（当 src 在路径中时）
        from utils import OUTPUT_DIR
        return OUTPUT_DIR
    except ImportError:
        try:
            # 尝试相对导入
            from .utils import OUTPUT_DIR
            return OUTPUT_DIR
        except ImportError:
            try:
                # 添加当前目录到路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                from utils import OUTPUT_DIR
                return OUTPUT_DIR
            except ImportError:
                # 最终备选方案
                print("⚠️ 无法导入 utils，使用备选路径")
                return "./data/example_output"

# 获取输出目录
OUTPUT_DIR = _import_utils()
print(f"📁 visualization.py 使用输出目录: {OUTPUT_DIR}")

def plot_interferogram(z, I_ideal, I_noisy, fname="interferogram.png"):
    """绘制干涉图"""
    plt.figure(figsize=(8,4))
    plt.plot(z*1e6, I_ideal, label="Ideal Signal")
    plt.plot(z*1e6, I_noisy, label="Noisy Signal", alpha=0.7)
    plt.xlabel("Scan Position (μm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("White Light Interferogram")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 干涉图已保存: {output_path}")
    plt.close()
    return output_path

def plot_phase(z, phase_fft, phase_hilbert, fname="phase_comparison.png"):
    """绘制相位对比图"""
    plt.figure(figsize=(8,4))
    plt.plot(z*1e6, phase_fft, label="FFT Phase")
    plt.plot(z*1e6, phase_hilbert, label="Hilbert Phase", alpha=0.7)
    plt.xlabel("Scan Position (μm)")
    plt.ylabel("Phase (rad)")
    plt.title("Phase Extraction Comparison")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 相位对比图已保存: {output_path}")
    plt.close()
    return output_path

def plot_surface(x, y, height_map, fname="reconstructed_surface.png"):
    """绘制三维表面图"""
    X, Y = np.meshgrid(y * 1e6, x)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, height_map * 1e9, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel("Scan Position (μm)")
    ax.set_ylabel("Line Index")
    ax.set_zlabel("Height (nm)")
    ax.set_title("Reconstructed Surface")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 重建表面图已保存: {output_path}")
    plt.close()
    return output_path