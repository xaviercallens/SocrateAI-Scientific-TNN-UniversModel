import numpy as np
import json
import time

def simulate_4f_fourier_correlator(spatial_resolution=128, defect_present=True):
    """
    Lab 1 Digital Twin: 4f Optical Correlator (Dual Space Fourier Transformation).
    Simulates spatial dual-space filtering (P1 pushed to P4 scale).
    """
    x = np.linspace(-5, 5, spatial_resolution)
    y = np.linspace(-5, 5, spatial_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Input object signal (grid / mesh)
    grid_signal = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    if defect_present:
        # Punctual localized defect (e.g. hair / pinhole perturbation)
        defect = np.exp(-((X - 1.0)**2 + (Y - 1.0)**2) / 0.05)
        grid_signal += defect
        
    # 2D Optical Fourier Transform (Dual Space at lens 1 focal plane)
    fourier_dual = np.fft.fftshift(np.fft.fft2(grid_signal))
    
    # High-pass filter in Dual Space (Pin spatial mask)
    R = np.sqrt(X**2 + Y**2)
    mask = (R > 0.5).astype(float)
    filtered_dual = fourier_dual * mask
    
    # Inverse FFT at lens 2 image plane (Reconstructed Real Space)
    reconstructed_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_dual)))
    
    return {
        "spatial_resolution": spatial_resolution,
        "dual_energy": float(np.sum(np.abs(fourier_dual)**2)),
        "reconstructed_energy": float(np.sum(reconstructed_image**2)),
        "defect_suppression_ratio": float(np.max(reconstructed_image) / (np.mean(reconstructed_image) + 1e-8))
    }

def get_lab1_electronic_schematics():
    """
    Returns electronic wiring plan for Lab-1 setup.
    """
    return {
        "lab": "LAB-1 (Optoelectronic 4f Correlator & Laser Profiler)",
        "components": [
            {"name": "5mW 532nm Green Laser", "pin": "VCC 5V / GND", "control": "MOSFET PWM via Pico GP10"},
            {"name": "BPW34 Photodiode Sensor", "pin": "Anode -> ADC0 (GP26), Cathode -> GND", "signal": "Analog Voltage"},
            {"name": "Photodiode Amplifier (OpAmp)", "pin": "LM358 Gain 100x", "power": "3.3V"}
        ],
        "protocols": ["Dual Space Optical FFT", "Single-point defect propagation"]
    }

if __name__ == "__main__":
    print("=== LAB-1 OPTICAL CORRELATOR DIGITAL TWIN ===")
    res = simulate_4f_fourier_correlator()
    print(f"[SIMULATION] Dual Energy: {res['dual_energy']:.2f} | Reconstructed Energy: {res['reconstructed_energy']:.2f}")
    schematics = get_lab1_electronic_schematics()
    print(f"[ELECTRONICS] Configured {len(schematics['components'])} hardware components for LAB-1.")
