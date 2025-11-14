# src/phase_unwrap.py
import numpy as np
from skimage.restoration import unwrap_phase as unwrap_2d

def unwrap_surface_2d(wrapped_phase_map):
    """
    对2D包裹相位图进行解包裹。
    """
    # skimage 的 unwrap_phase 是一个强大的 2D 解包裹算法 [43, 44, 45, 46, 47]
    print("...开始 2D 相位解包裹 (可能需要几秒钟)...")
    unwrapped_surface = unwrap_2d(wrapped_phase_map)
    print("...2D 解包裹完成。")
    return unwrapped_surface