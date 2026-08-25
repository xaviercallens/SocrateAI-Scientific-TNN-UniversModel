#!/usr/bin/env python3
"""
LAB-6 EXTENSION: Numerical Verification of the Galerkin Energy Estimate
========================================================================
PURPOSE: Bridge between the Lean 4 formalization (certs/NS_Galerkin_Energy.lean)
         and the LAB-6 FGRS Oracle (scripts/lab6_fgrs_oracle.py).

WHAT THIS PROVES NUMERICALLY:
  1. Triadic antisymmetry: ⟨B(u,u), u⟩ = 0  (energy conservation by nonlinearity)
  2. Dissipation monotonicity: E(t+dt) ≤ E(t)  (viscosity only removes energy)
  3. Poincaré lower bound: D ≥ ν·λ₁·E  (exponential decay rate)
  4. Divergence freedom: k · û_k = 0  (Leray projection exact)

THESE NUMERICS VALIDATE that the Lean 4 theorems apply to the LAB-6 engine.
They do NOT replace the formal proofs — they confirm the implementation matches.

CONNECTS TO:
  - certs/NS_Galerkin_Energy.lean: formal proofs of theorems verified here
  - scripts/lab6_fgrs_oracle.py: the triadic engine being validated
  - specs/SPECIFICATION MEMO: LAB-6 Navier Spoke.md: the FGRS architecture
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import json
import datetime
import hashlib
from pathlib import Path

from scripts.lab6_fgrs_oracle import (
    build_truncated_lattice,
    init_divergence_free_field,
    compute_exact_triadic_derivative,
    leray_project
)

# =====================================================================
# 1. ENERGY FUNCTIONAL (mirrors Lean 4: energy = Σ |û_k|²)
# =====================================================================

def compute_energy(u_hat: dict) -> float:
    """Total kinetic energy: E = Σ_k |û_k|²."""
    return sum(np.sum(np.abs(v)**2) for v in u_hat.values())


def compute_enstrophy(u_hat: dict) -> float:
    """Enstrophy: Ω = Σ_k |k|² |û_k|²."""
    return sum(
        (k[0]**2 + k[1]**2 + k[2]**2) * np.sum(np.abs(v)**2)
        for k, v in u_hat.items()
    )


def compute_dissipation(u_hat: dict, nu: float) -> float:
    """Viscous dissipation: D = 2ν · Ω = 2ν Σ_k |k|² |û_k|²."""
    return 2 * nu * compute_enstrophy(u_hat)


# =====================================================================
# 2. VERIFICATION 1: TRIADIC ANTISYMMETRY ⟨B(u,u), u⟩ = 0
# =====================================================================

def verify_triadic_antisymmetry(M: int = 4, n_trials: int = 10, tol: float = 1e-10) -> dict:
    """
    The nonlinear term B(u,u) in Navier-Stokes CONSERVES energy:
        Re⟨B(u,u), u⟩ = Re Σ_k conj(û_k) · (∂_t û_k)_nonlinear = 0
    
    This identity makes the Galerkin energy estimate E' = -2νΩ work.
    Without it, the nonlinearity could inject or remove energy.
    
    This is the COMPUTATIONAL VALIDATION of the premise used in 
    NS_Galerkin_Energy.lean:energy_decreases_under_dissipation.
    """
    modes = build_truncated_lattice(M)
    max_inner_product = 0.0
    results = []
    
    for trial in range(n_trials):
        u_hat = init_divergence_free_field(modes, seed=trial * 7 + 13)
        
        # Compute the nonlinear derivative (no viscosity term)
        du_dt = compute_exact_triadic_derivative(u_hat, modes)
        
        # Inner product: Re⟨du/dt, u⟩ = Re Σ_k conj(û_k) · (∂_t û_k)
        inner_product = 0.0
        for k in modes:
            inner_product += np.real(np.dot(np.conj(u_hat[k]), du_dt[k]))
        
        E = compute_energy(u_hat)
        relative = abs(inner_product) / max(E, 1e-30)
        max_inner_product = max(max_inner_product, abs(inner_product))
        
        results.append({
            "trial": trial,
            "energy": float(E),
            "inner_product_abs": float(abs(inner_product)),
            "relative_error": float(relative)
        })
    
    passed = max_inner_product < tol
    
    print(f"  TRIADIC ANTISYMMETRY: max |⟨B(u,u),u⟩| = {max_inner_product:.2e}")
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} (tolerance: {tol:.0e})")
    
    return {
        "test": "triadic_antisymmetry",
        "lean4_theorem": "energy_decreases_under_dissipation (premise validation)",
        "M": M,
        "n_trials": n_trials,
        "max_inner_product": float(max_inner_product),
        "tolerance": tol,
        "passed": passed,
        "trials": results
    }


# =====================================================================
# 3. VERIFICATION 2: ENERGY MONOTONICITY UNDER DISSIPATION
# =====================================================================

def verify_energy_monotonicity(M: int = 3, nu: float = 0.1, n_steps: int = 20,
                                dt: float = 0.001) -> dict:
    """
    Explicit Euler integration of the Galerkin NS system:
        û_k(t+dt) = û_k(t) + dt · [B(u,u)_k - ν|k|² û_k]
    
    The energy E(t) = Σ|û_k|² must be monotonically decreasing
    (to within O(dt²) Euler truncation error).
    
    This validates NS_Galerkin_Energy.lean:energy_decreases_under_dissipation.
    """
    modes = build_truncated_lattice(M)
    u_hat = init_divergence_free_field(modes, seed=42, energy_scale=0.5)
    
    energies = []
    enstrophies = []
    
    for step in range(n_steps):
        E = compute_energy(u_hat)
        Omega = compute_enstrophy(u_hat)
        energies.append(E)
        enstrophies.append(Omega)
        
        # Compute nonlinear derivative
        du_dt = compute_exact_triadic_derivative(u_hat, modes)
        
        # Add viscous damping: -ν|k|² û_k
        for k in modes:
            k_sq = k[0]**2 + k[1]**2 + k[2]**2
            du_dt[k] -= nu * k_sq * u_hat[k]
        
        # Euler step
        for k in modes:
            u_hat[k] = u_hat[k] + dt * du_dt[k]
        
        # Re-project to enforce divergence freedom (Euler can drift)
        for k in modes:
            k_arr = np.array(k, dtype=np.float64)
            u_hat[k] = leray_project(k_arr, u_hat[k])
    
    # Check monotonicity
    monotone_violations = 0
    max_violation = 0.0
    for i in range(1, len(energies)):
        if energies[i] > energies[i-1] + 1e-12:  # Allow tiny Euler error
            monotone_violations += 1
            max_violation = max(max_violation, energies[i] - energies[i-1])
    
    passed = monotone_violations == 0
    decay_ratio = energies[-1] / energies[0] if energies[0] > 0 else 0
    
    print(f"  ENERGY MONOTONICITY: E(0)={energies[0]:.6f} → E({n_steps})={energies[-1]:.6f}")
    print(f"  Decay ratio: {decay_ratio:.4f}, violations: {monotone_violations}")
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    
    return {
        "test": "energy_monotonicity",
        "lean4_theorem": "energy_decreases_under_dissipation",
        "M": M,
        "nu": nu,
        "dt": dt,
        "n_steps": n_steps,
        "E_initial": float(energies[0]),
        "E_final": float(energies[-1]),
        "decay_ratio": float(decay_ratio),
        "monotone_violations": monotone_violations,
        "max_violation": float(max_violation),
        "passed": passed
    }


# =====================================================================
# 4. VERIFICATION 3: POINCARÉ LOWER BOUND ON DISSIPATION
# =====================================================================

def verify_poincare_bound(M: int = 4, nu: float = 0.01, n_trials: int = 10) -> dict:
    """
    Poincaré inequality for the Galerkin system:
        D = 2ν Σ |k|²|û_k|² ≥ 2ν λ₁ Σ |û_k|² = 2ν λ₁ E
    
    where λ₁ = min{|k|² : k ∈ lattice, k ≠ 0} = 1 (for standard Z³ lattice).
    
    This validates NS_Galerkin_Energy.lean:poincare_dissipation_bound.
    """
    modes = build_truncated_lattice(M)
    
    # λ₁ = smallest |k|² in the lattice (should be 1 for Z³)
    lambda_1 = min(k[0]**2 + k[1]**2 + k[2]**2 for k in modes)
    
    violations = 0
    min_ratio = float('inf')
    
    for trial in range(n_trials):
        u_hat = init_divergence_free_field(modes, seed=trial * 13 + 7)
        E = compute_energy(u_hat)
        D = compute_dissipation(u_hat, nu)
        
        # Poincaré bound: D ≥ 2ν λ₁ E
        lower_bound = 2 * nu * lambda_1 * E
        ratio = D / max(lower_bound, 1e-30)
        min_ratio = min(min_ratio, ratio)
        
        if D < lower_bound - 1e-12:
            violations += 1
    
    passed = violations == 0
    
    print(f"  POINCARÉ BOUND: λ₁={lambda_1}, min(D/(2νλ₁E))={min_ratio:.6f}")
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} ({violations} violations in {n_trials} trials)")
    
    return {
        "test": "poincare_bound",
        "lean4_theorem": "poincare_dissipation_bound",
        "M": M,
        "lambda_1": int(lambda_1),
        "nu": nu,
        "n_trials": n_trials,
        "min_ratio_D_over_bound": float(min_ratio),
        "violations": violations,
        "passed": passed
    }


# =====================================================================
# 5. VERIFICATION 4: DIVERGENCE FREEDOM (LERAY PROJECTOR)
# =====================================================================

def verify_divergence_freedom(M: int = 4, n_trials: int = 10, tol: float = 1e-14) -> dict:
    """
    Every initialized field must satisfy k · û_k = 0 ∀k.
    This is the divergence-free condition in Fourier space, enforced
    by the Leray projector.
    
    Validates NS_Galerkin_Energy.lean:inner_sub_proj_eq_zero.
    """
    modes = build_truncated_lattice(M)
    max_div = 0.0
    
    for trial in range(n_trials):
        u_hat = init_divergence_free_field(modes, seed=trial * 5 + 3)
        
        for k in modes:
            k_arr = np.array(k, dtype=np.float64)
            div = abs(np.dot(k_arr, u_hat[k]))
            max_div = max(max_div, div)
    
    passed = max_div < tol
    
    print(f"  DIVERGENCE FREEDOM: max |k·û_k| = {max_div:.2e}")
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} (tolerance: {tol:.0e})")
    
    return {
        "test": "divergence_freedom",
        "lean4_theorem": "inner_sub_proj_eq_zero",
        "M": M,
        "n_trials": n_trials,
        "max_divergence": float(max_div),
        "tolerance": tol,
        "passed": passed
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 74)
    print(" LAB-6 EXTENSION: Galerkin Energy Estimate — Numerical Verification")
    print(" Bridge: NS_Galerkin_Energy.lean ↔ lab6_fgrs_oracle.py")
    print("=" * 74)
    
    out_dir = Path("certs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Test 1: Triadic Antisymmetry
    print("\n" + "─" * 74)
    print("TEST 1: Triadic Antisymmetry ⟨B(u,u), u⟩ = 0")
    print("  Lean 4: energy_decreases_under_dissipation (premise)")
    print("─" * 74)
    results.append(verify_triadic_antisymmetry(M=4, n_trials=10))
    
    # Test 2: Energy Monotonicity
    print("\n" + "─" * 74)
    print("TEST 2: Energy Monotonicity under Viscous Dissipation")
    print("  Lean 4: energy_decreases_under_dissipation")
    print("─" * 74)
    results.append(verify_energy_monotonicity(M=3, nu=0.1, n_steps=20))
    
    # Test 3: Poincaré Lower Bound
    print("\n" + "─" * 74)
    print("TEST 3: Poincaré Dissipation Lower Bound D ≥ 2νλ₁E")
    print("  Lean 4: poincare_dissipation_bound")
    print("─" * 74)
    results.append(verify_poincare_bound(M=4, nu=0.01, n_trials=10))
    
    # Test 4: Divergence Freedom
    print("\n" + "─" * 74)
    print("TEST 4: Divergence Freedom k · û_k = 0")
    print("  Lean 4: inner_sub_proj_eq_zero")
    print("─" * 74)
    results.append(verify_divergence_freedom(M=4, n_trials=10))
    
    # Summary
    print("\n" + "=" * 74)
    print(" SUMMARY")
    print("=" * 74)
    all_passed = True
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} {r['test']:30s} → Lean 4: {r['lean4_theorem']}")
        if not r["passed"]:
            all_passed = False
    
    print(f"\n  OVERALL: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    
    # Certification
    cert = {
        "pipeline": "LAB-6 GALERKIN ENERGY NUMERICAL VERIFICATION",
        "purpose": "Bridge between NS_Galerkin_Energy.lean and lab6_fgrs_oracle.py",
        "timestamp": datetime.datetime.now().isoformat(),
        "all_passed": all_passed,
        "tests": [{k: v for k, v in r.items() if k != "trials"} for r in results],
        "lean4_file": "certs/NS_Galerkin_Energy.lean",
        "oracle_file": "scripts/lab6_fgrs_oracle.py",
        "status": "TIER_A_NUMERICAL_BRIDGE_CERTIFIED" if all_passed else "FAILURES"
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)
    
    cert_path = out_dir / "lab6_galerkin_energy_verification.json"
    with open(cert_path, "w") as f:
        json.dump(cert, f, indent=4, cls=NumpyEncoder)
    
    print(f"\n[SUCCESS] Certification saved: {cert_path}")


if __name__ == "__main__":
    main()
