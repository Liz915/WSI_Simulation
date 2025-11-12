# main.py
import numpy as np
from signal_generator import simulate_wsi_signal
from src.noise_model import add_noise
from src.fft_phase_extraction import extract_phase_fft
from src.hilbert_phase_extraction import extract_phase_hilbert
from src.phase_unwrap import unwrap_phase
from src.visualization import plot_interferogram, plot_phase, plot_surface

def main():
    print("🚀 Starting WSI Signal Simulation and Reconstruction...")

    # Step 1. 生成扫描轴
    z = np.linspace(-2e-6, 2e-6, 2000)

    # Step 2. 模拟WSI信号
    I_ideal = simulate_wsi_signal(z, lambda_c=600e-9, Lc=0.8e-6, h0=200e-9)
    print("✅ Generated interferogram")

    # Step 3. 加入噪声（机械振动 + 探测器噪声）
    I_noisy = add_noise(I_ideal, z, vib_amp=20e-9, vib_freq=30, snr_db=25)
    print("✅ Noise injected")

    # Step 4. FFT提取相位
    phase_fft = extract_phase_fft(I_noisy, z)
    print("✅ Phase extracted by FFT method")

    # Step 5. Hilbert提取相位
    phase_hilbert = extract_phase_hilbert(I_noisy)
    print("✅ Phase extracted by Hilbert method")

    # Step 6. 解包裹
    phase_unwrapped = unwrap_phase(phase_fft)
    print("✅ Phase unwrapped")

    # Step 7. 可视化
    plot_interferogram(z, I_ideal, I_noisy)
    plot_phase(z, phase_fft, phase_hilbert)
    height_map = np.tile(phase_unwrapped, (50, 1)) * (600e-9 / (4 * np.pi))
    plot_surface(np.arange(height_map.shape[0]), z, height_map)

    print("🎉 All steps completed. Check figures in output.")

if __name__ == "__main__":
    main()