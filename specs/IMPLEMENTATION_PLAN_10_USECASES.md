# TNN Univers Model — Implementation Plan: 10 Physics Use Cases
**Author**: SocrateAI Lab  
**Date**: 2026-08-14  
**Status**: IN PROGRESS

---

## Governing Invariants (Non-Negotiable)

| Rule | Mandate |
|:---|:---|
| **Zero-Stub** | `torch.randn` is BANNED for benchmark/training data. All ICs must be deterministic (seeded or analytical). |
| **Real / Ab-Initio Data** | MD17 and QM9 use real DFT trajectories. PDE use cases use spectrally-exact solutions or publicly available datasets. |
| **Symplectic Integrator** | All rollout evaluations use RK4 or Störmer-Verlet. Euler is BANNED for energy drift measurement. |
| **Scientific Audit** | Each script appends a timestamped certificate to `specs/Scientific_Audit_Ledger.md`. |
| **Equivariance Gate** | All use cases with 3D geometry run the `EquivarianceHook` before training. |
| **Convergence** | Training must show monotonic loss decrease over ≥ 100 epochs with final test loss < 1e-2. |

---

## Architecture per Use Case

Each script follows the same TNN template:

```
1. Data Generator (deterministic / real DFT)
      ↓
2. TNN Architecture (EGNN | HNN | FNO — depending on pillar)
      ↓
3. Training Loop (Adam, 100+ epochs, batched)
      ↓
4. Zero-Sorry Verification Hook
      ↓
5. Rollout Benchmark (RK4 / Störmer-Verlet)
      ↓
6. Audit Certificate → Scientific_Audit_Ledger.md
```

---

## Use Case Inventory

| # | Domain | Script | Pillar | Model | Data Source | Status |
|:--|:---|:---|:---|:---|:---|:---|
| 1 | Mécanique Classique | `train_usecase_spring.py` | Thermo | HNN | Deterministic spring ICs | ✅ EXISTING |
| 2 | Astrophysique | `train_usecase_3body_gravitation.py` | Topo+Thermo | EGNN+HNN+RK4 | Seeded RK4 integration | ✅ EXISTING |
| 3 | Électromagnétisme | `train_usecase_lorentz.py` | Thermo | HNN + Vector Field | Cyclotron orbit (analytical) | 🔴 TODO |
| 4 | Dynamique Non-Linéaire | `train_usecase_double_pendulum.py` | Thermo | Non-linear HNN | Double pendulum Hamiltonian | 🔴 TODO |
| 5 | Thermodynamique Stat. | `train_usecase_gas_kinetics.py` | Thermo | Energy Critic | Maxwell-Boltzmann distribution | 🔴 TODO |
| 6 | Physique Quantique | `train_usecase_schrodinger.py` | Tensor | Complex HNN / FNO | Coherent state wavepackets | 🔴 TODO |
| 7 | Mécanique des Fluides | `train_usecase_burgers.py` | Tensor | FNO 1D | Burgers shockwaves (deterministic) | 🔴 TODO |
| 8 | Relativité Restreinte | `train_usecase_relativistic.py` | Thermo | Relativistic HNN | Relativistic Hamiltonian (analytical) | 🔴 TODO |
| 9 | Électrodynamique | `train_usecase_wave_equation.py` | Tensor | FNO / Wave Operator | D'Alembert traveling waves | 🔴 TODO |
| 10 | Cosmologie | `train_usecase_cosmo_flrw.py` | Thermo | Friedmann HNN | FLRW exact solution | 🔴 TODO |

---

## Implementation Phases

### Phase A — Thermodynamic Pillar Use Cases (HNN-based)
Scripts: UC3 (Lorentz), UC4 (Double Pendulum), UC5 (Gas Kinetics), UC8 (Relativistic), UC10 (FLRW)

**Common Architecture**:
```python
class TNNHamiltonianModel(nn.Module):
    # Learns H(q, p) as a scalar
    # Hamilton's equations via torch.autograd
    # dq/dt = dH/dp,  dp/dt = -dH/dq
```

**Verification Hook**: `SymplecticConservationHook` — ΔH over 500 RK4 steps < 1e-2

### Phase B — Tensor Pillar Use Cases (FNO-based)
Scripts: UC6 (Schrödinger), UC7 (Burgers), UC9 (Wave Equation)

**Common Architecture**:
```python
class TNNFourierOperator(nn.Module):
    # FNO learns the solution operator u_t -> u_{t+1}
    # Mesh-free: resolution-independent prediction
```

**Verification Hook**: `MassConservationHook` — divergence < 1e-2 (fluids), L2 norm conservation (quantum)

---

## Validation Criteria (per use case)

| Metric | Pass Threshold | Measurement Method |
|:---|:---|:---|
| Train Loss | < 1e-2 final | MSE or L1 over last 10 epochs |
| Test Loss | < 1e-2 | Held-out 20% split |
| Energy Drift (Hamiltonian) | < 5% of H_0 | 500-step RK4 rollout |
| Equivariance Violation | < 1e-4 | EquivarianceHook (EGNN cases only) |
| L2 Norm Drift (PDE) | < 10% | 100-step FNO rollout |
| Audit Certificate | Present | Appended to Ledger automatically |

---

## Deliverables Checklist

- [ ] UC3 `train_usecase_lorentz.py` — implemented & ✅ PASS
- [ ] UC4 `train_usecase_double_pendulum.py` — implemented & ✅ PASS
- [ ] UC5 `train_usecase_gas_kinetics.py` — implemented & ✅ PASS
- [ ] UC6 `train_usecase_schrodinger.py` — implemented & ✅ PASS
- [ ] UC7 `train_usecase_burgers.py` — implemented & ✅ PASS
- [ ] UC8 `train_usecase_relativistic.py` — implemented & ✅ PASS
- [ ] UC9 `train_usecase_wave_equation.py` — implemented & ✅ PASS
- [ ] UC10 `train_usecase_cosmo_flrw.py` — implemented & ✅ PASS
- [ ] `run_all_10_usecases.py` — master runner with summary table
- [ ] README updated with verified results
- [ ] `Scientific_Audit_Ledger.md` updated with all 10 certificates
