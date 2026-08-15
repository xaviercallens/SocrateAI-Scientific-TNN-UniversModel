import numpy as np

class CHOPVortexObservatory:
    """
    Lab 3: White Hole 1D Channel & Draining Vortex CHOP Observatory.
    Simulates Froude Fr=1 horizon profiling, optical flow elevation & Bernoulli hydraulic flow.
    """
    def __init__(self, channel_length=1.0, num_points=128):
        self.length = channel_length
        self.num_points = num_points
        self.x = np.linspace(-channel_length/2, channel_length/2, num_points)

    def get_electronic_schematics(self):
        return {
            "lab": "LAB-3 (CHOP Hydrodynamic Observatory)",
            "components": [
                {"name": "12V Submersible Circulation Pump (1500L/h)", "control": "IRLZ44N MOSFET PWM via Pico GP20", "flyback_diode": "1N4007"},
                {"name": "5V UV Perturber Solenoid", "control": "TIP120 Transistor via Pico GP21"},
                {"name": "FCD Chequerboard Backlight Panel", "power": "12V LED Grid"},
                {"name": "Overhead High-Speed Camera", "interface": "USB Video Stream to Host (OpenCV FCD)"}
            ],
            "fluid_metric": "Fast-Chequerboard Demodulation 3D Surface Reconstruction"
        }

    def simulate_hydrodynamic_horizon(self, pump_pwm=1.0, obstacle_height=0.06):
        """
        Simulates 1D Rousseaux obstacle flow / drain vortex horizon profile.
        Calculates local water depth h(x), fluid velocity v(x), wave celerity c(x), and Froude number Fr(x).
        """
        g = 9.81
        h0 = 0.10 # Baseline undisturbed water depth
        
        # Topographic obstacle (Gaussian bump)
        obstacle = obstacle_height * np.exp(-self.x**2 / 0.02)
        h = h0 - obstacle # Local water depth
        
        # Mass conservation: Q = v(x) * h(x) -> v(x) = Q / h(x)
        base_Q = 0.08 * pump_pwm # Discharge rate scaled by pump PWM
        v = base_Q / h
        
        # Shallow water wave celerity c = sqrt(g * h)
        c = np.sqrt(g * h)
        
        # Froude Number Fr = v / c
        Froude = v / c
        
        # Horizon location where Fr = 1
        horizon_indices = np.where(np.diff(np.sign(Froude - 1.0)))[0]
        rh_coords = [float(self.x[idx]) for idx in horizon_indices]
        
        return {
            "x_coords": self.x.tolist(),
            "froude_profile": Froude.tolist(),
            "velocity_profile": v.tolist(),
            "celerity_profile": c.tolist(),
            "horizon_coords_rh": rh_coords,
            "is_horizon_present": len(rh_coords) > 0,
            "max_froude": float(np.max(Froude))
        }

if __name__ == "__main__":
    print("=== LAB-3 CHOP HYDRODYNAMIC OBSERVATORY SIMULATOR ===")
    chop = CHOPVortexObservatory()
    schem = chop.get_electronic_schematics()
    print(f"[ELECTRONICS] Pump PWM MOSFET + Solenoid + FCD Camera configured.")
    sim_res = chop.simulate_hydrodynamic_horizon(pump_pwm=0.85)
    print(f"[HORIZON SIMULATION] Horizon Present: {sim_res['is_horizon_present']} | Max Froude: {sim_res['max_froude']:.3f}")
    if sim_res['is_horizon_present']:
        print(f"[HORIZON COORDINATE] r_h identified at x = {sim_res['horizon_coords_rh']}")
