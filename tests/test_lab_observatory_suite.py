import pytest
import numpy as np
import torch
import hashlib
import json
import sys
import os

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from lab1_optics_correlator_sim import simulate_4f_fourier_correlator, get_lab1_electronic_schematics
from lab2_boma2d_driver_sim import BOMA2DController
from lab3_chop_vortex_sim import CHOPVortexObservatory
from hardware_dshl_epistemological_lock import epistemological_lock
from tnn_holographic_p4_detector import HolographicTNN, detect_p4_algorithmic_crash, execute_no_magic_theorem


def test_lab0_epistemological_lock():
    """LAB-0 UNIT TEST: Pre-registered prediction SHA-256 lock & hashing."""
    config = {"lab": "LAB-0", "mode": "baseline_test"}
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    lock_hash = epistemological_lock(config, prediction)
    
    assert isinstance(lock_hash, str)
    assert len(lock_hash) == 64  # SHA-256 hex string length
    
    # Verify file created
    lock_file = f"LOCKED_RUN_{lock_hash[:8]}.meta"
    assert os.path.exists(lock_file)
    os.remove(lock_file)


def test_lab1_4f_correlator():
    """LAB-1 UNIT TEST: Dual Space Optical FFT Correlator & Defect Suppression."""
    result = simulate_4f_fourier_correlator(spatial_resolution=64, defect_present=True)
    
    assert "defect_suppression_ratio" in result
    assert result["defect_suppression_ratio"] > 1.0
    assert result["dual_energy"] > 0
    assert result["reconstructed_energy"] > 0


def test_lab2_boma2d_scanner():
    """LAB-2 UNIT TEST: BOMA-2D Dual-Axis Stepper Scanner & Photodiode ADC Matrix."""
    boma = BOMA2DController(x_steps=16, y_steps=16)
    schematics = boma.get_electronic_schematics()
    scan_res = boma.simulate_dual_space_scan()
    
    assert schematics["lab"].startswith("LAB-2")
    assert len(schematics["components"]) == 3
    assert scan_res["grid_size"] == [16, 16]
    assert 0.0 <= scan_res["max_intensity"] <= 1.0


def test_lab3_chop_vortex_horizon():
    """LAB-3 UNIT TEST: Hydrodynamic Froude Profile & Transcritical Horizon Fr=1.0."""
    chop = CHOPVortexObservatory(num_points=64)
    res = chop.simulate_hydrodynamic_horizon(pump_pwm=1.2, obstacle_height=0.06)
    
    assert res["is_horizon_present"] is True
    assert res["max_froude"] > 1.0
    assert len(res["horizon_coords_rh"]) >= 1


def test_lab4_tnn_holographic_mapping():
    """LAB-4 UNIT TEST: TNN Boundary-to-Bulk Area Law Bottleneck (chi=8)."""
    boundary_dim = 16
    bond_dim = 8
    bulk_res = 32
    batch_size = 10
    
    tnn = HolographicTNN(boundary_dim=boundary_dim, bond_dimension_chi=bond_dim, bulk_resolution=bulk_res)
    boundary_tensor = torch.randn(batch_size, boundary_dim)
    
    bulk_pred = tnn(boundary_tensor)
    assert bulk_pred.shape == (batch_size, bulk_res, bulk_res)


def test_lab4_no_magic_theorem_negative_control():
    """LAB-4 UNIT TEST: No-Magic Theorem (Negative Control / Brisure de Jauge)."""
    boundary_dim = 16
    bond_dim = 8
    bulk_res = 32
    batch_size = 10
    
    tnn = HolographicTNN(boundary_dim=boundary_dim, bond_dimension_chi=bond_dim, bulk_resolution=bulk_res)
    
    baseline_error = 2.0
    mock_boundary_turbulent = torch.randn(batch_size, boundary_dim) * 2.0
    mock_real_bulk_turbulent = torch.randn(batch_size, bulk_res, bulk_res) * 5.0
    
    turbulent_error = execute_no_magic_theorem(tnn, mock_boundary_turbulent, mock_real_bulk_turbulent, baseline_error)
    
    # Assert that chaotic turbulence causes error explosion (> 5x baseline)
    assert turbulent_error > baseline_error * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
