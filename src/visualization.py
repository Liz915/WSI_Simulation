# src/visualization.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- 这是修复了 Bug 的 Cell 2 函数 ---
def plot_stack_section(stack, title="WSI stack section", Z_SCAN=None):
    """
    显示 WSI 信号栈在 z-x 的切片（取中间 y 行）。
    stack: (n_z, n_y, n_x)
    """
    n_z, n_y, n_x = stack.shape # 需要先获取 x 轴的 'n_x'
    plt.figure(figsize=(10, 4))
    
    # 根据 Z_SCAN 范围设置 y 轴刻度
    if Z_SCAN is not None:
        # 修复后的 extent 格式: [left, right, bottom, top]
        # x轴: 从第0个像素到最后一个像素
        left = 0
        right = n_x - 1
        # y轴: Z-scan 的实际物理位置 (转换为微米)
        bottom = Z_SCAN[0] * 1e6    # 起始位置 (μm)
        top = Z_SCAN[-1] * 1e6      # 结束位置 (μm)
        
        extent = [left, right, bottom, top]
        plt.imshow(stack[:, n_y//2, :], aspect='auto', cmap='viridis', origin='lower', extent=extent)
        plt.ylabel("Z-scan Position (μm)")
    else:
        plt.imshow(stack[:, n_y//2, :], aspect='auto', cmap='viridis', origin='lower')
        plt.ylabel("Z-scan index")
        
    plt.title(title)
    plt.xlabel("X Pixel")
    plt.colorbar(label="Intensity (a.u.)")
    plt.show() # 在 Notebook 中使用 plt.show()

# --- 你的其他绘图函数 ---

def get_output_dir():
    """ 智能获取和创建输出目录 """
    # 假设 Notebook 在 'notebooks/' 文件夹中
    proj_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if 'notebooks' not in proj_root:
        proj_root = os.getcwd() # 备用方案
        
    output_dir = os.path.join(proj_root, "data", "example_output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

OUTPUT_DIR = get_output_dir()

def plot_interferogram(z, I_ideal, I_noisy, fname="interferogram.png"):
    """绘制干涉图"""
    plt.figure(figsize=(8,4))
    plt.plot(z*1e6, I_ideal, 'b--', label="理想信号", alpha=0.7)
    plt.plot(z*1e6, I_noisy, 'r-', label="含噪信号", alpha=0.8)
    plt.xlabel("Scan Position (μm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("White Light Interferogram")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 干涉图已保存: {output_path}")
    plt.show() # 在 Notebook 中使用 plt.show()

def plot_surface(x_pixels, y_pixels, height_map, fname="reconstructed_surface.png"):
    """绘制三维表面图"""
    X, Y = np.meshgrid(x_pixels, y_pixels)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 限制显示范围，突出台阶特征
    display_data = np.clip(height_map, 
                           np.percentile(height_map, 1), 
                           np.percentile(height_map, 99))
    
    surf = ax.plot_surface(X, Y, display_data, cmap='viridis', linewidth=0, antialiased=False, rstride=3, cstride=3)
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")
    ax.set_zlabel("Height (nm)")
    ax.set_title("Reconstructed Surface")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 重建表面图已保存: {output_path}")
    plt.show() # 在 Notebook 中使用 plt.show()