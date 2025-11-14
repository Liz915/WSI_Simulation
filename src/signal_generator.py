# src/signal_generator.py
import numpy as np

def create_simulated_surface(shape=(128,128), step_height_nm=40.0):
    """
    创建一个包含一个纳米级台阶的模拟2D表面。
    """
    surface = np.zeros(shape)
    sy, sx = shape
    step_start_y, step_start_x = sy//4, sx//4
    step_end_y, step_end_x = sy*3//4, sx*3//4
    surface[step_start_y:step_end_y, step_start_x:step_end_x] = step_height_nm * 1e-9 # 转换为米
    return surface

def simulate_wsi_stack_3d(surface, z_scan, lambda_c=600e-9, Lc=0.8e-6, Idc=1.0, A0=0.6):
    """
    为整个2D表面生成3D WSI信号栈 (z, y, x)。
    """
    n_z = len(z_scan)
    n_y, n_x = surface.shape
    
    # 使用 NumPy broadcasting 来生成3D信号栈
    z_3d = z_scan.reshape(n_z, 1, 1)
    surface_3d = surface.reshape(1, n_y, n_x)
    
    opd = z_3d - surface_3d
    envelope = A0 * np.exp(- (opd / Lc)**2) 
    phase = (4 * np.pi / lambda_c) * opd
    stack = Idc + envelope * np.cos(phase)
    
    return stack