import numpy as np
import time

class BOMA2DController:
    """
    Lab 2: Automated 2D Opto-Mechanical Scanner (BOMA-2D).
    Controls Dual-axis DVD stepper carriages and ADC photodiode reader.
    """
    def __init__(self, x_steps=50, y_steps=50, step_microns=10):
        self.x_steps = x_steps
        self.y_steps = y_steps
        self.step_microns = step_microns
        self.position = [0, 0]

    def get_electronic_schematics(self):
        return {
            "lab": "LAB-2 (BOMA-2D Dual-Axis Stepper Scanner)",
            "components": [
                {"name": "X-Axis DVD Stepper Motor", "driver": "A4988 #1", "pins": "STEP: Pico GP14, DIR: Pico GP15", "power": "12V VMOT"},
                {"name": "Y-Axis DVD Stepper Motor", "driver": "A4988 #2", "pins": "STEP: Pico GP16, DIR: Pico GP17", "power": "12V VMOT"},
                {"name": "ADS1115 16-Bit ADC", "interface": "I2C0 (SDA: GP0, SCL: GP1)", "address": "0x48", "channel": "A0 (BPW34 Light Intensity)"}
            ],
            "microstepping": "1/16 step mode for sub-micrometer resolution"
        }

    def simulate_dual_space_scan(self, k_center=0.0):
        """
        Simulates 2D raster scan of the dual/Fourier plane light intensity.
        """
        intensity_map = np.zeros((self.x_steps, self.y_steps))
        kx = np.linspace(-3, 3, self.x_steps)
        ky = np.linspace(-3, 3, self.y_steps)
        KX, KY = np.meshgrid(kx, ky)
        
        # Airy disk pattern / Speckle correlation profile in Fourier space
        R = np.sqrt((KX - k_center)**2 + KY**2)
        intensity_map = (np.sin(R + 1e-5) / (R + 1e-5))**2
        
        return {
            "grid_size": [self.x_steps, self.y_steps],
            "max_intensity": float(np.max(intensity_map)),
            "min_intensity": float(np.min(intensity_map)),
            "mean_intensity": float(np.mean(intensity_map)),
            "data_matrix": intensity_map.tolist()
        }

if __name__ == "__main__":
    print("=== LAB-2 BOMA-2D AUTOMATED SCANNER DRIVER & SIMULATOR ===")
    boma = BOMA2DController(x_steps=32, y_steps=32)
    schem = boma.get_electronic_schematics()
    print(f"[ELECTRONICS] Dual A4988 Steppers + ADS1115 I2C configuration ready.")
    scan_res = boma.simulate_dual_space_scan()
    print(f"[SCAN COMPLETED] 32x32 raster scan simulated. Max Intensity: {scan_res['max_intensity']:.4f}")
