# Naming Policy — SocrateAI-Scientific-TNN-UniversModel

**Effective:** 2026-08-25  
**Rule:** RES-1 (Reserved Scientific Terms) + N-1 (Structural Claims)  
**Status:** ENFORCED — violations in this repo block publication

---

## 1. Reserved Terms

The following terms carry formal mathematical meaning and **MUST NOT** appear as architecture or brand labels unless the code contains a proven implementation:

| Term | Definition Required | Where It Currently Applies |
|---|---|---|
| **Topological** | Verified TDA (persistent homology, Betti numbers, etc.) or proven invariant under homeomorphism | Lab5 TDA pipeline (`lab5_tda_holographic_clustering.py`), `topological_hash.py` (Euler characteristic). **NOT** Lab1 ResConv1D. |
| **Poly-Algebraic Calculus** | $N$-arity hyper-variable $\Xi^{\langle N \rangle}$ formalism as specified in `specs/Foundations of Poly-Algebraic Calculus.md` | Foundational spec documents, vHPU emulator (Rulial Invert concept). **NOT** dashboard labels or UI titles. |
| **Hamiltonian** | Conserved quantity of a symplectic flow ($\dot{q} = \partial H / \partial p$, $\dot{p} = -\partial H / \partial q$) | `tnn/physics/symplectic_fno.py` (actual HNN systems). **NOT** advection-diffusion proxy L2-norm. |
| **Topological TNN** | — | **BANNED** — No architecture in this repo carries proven topological structure per N-1. Use `ResConv1D (residual baseline)` for the 1D convolutional model. |

---

## 2. "TNN" in the Repository Name

The acronym TNN stands for **Thermodynamic, Topological, Tensor Neural Network**. This describes the *aspirational full architecture* (3 pillars), not any single benchmark model.

- **Thermodynamic pillar**: Hamiltonian / Lagrangian Neural Networks → implemented in use cases 1–10 (spring, 3-body, double pendulum, etc.)
- **Topological pillar**: E(n)-equivariant GNNs (EGNN/e3nn) + TDA → implemented in Lab5, EGNN use cases. **NOT implemented** in Lab1 advection-diffusion.
- **Tensor pillar**: FNO / Modulus PDE operators → implemented in Burgers, wave equation use cases.

Any benchmark that does not use all 3 pillars **MUST** name only the specific architecture under test (e.g., "ResConv1D", "Baseline CNN 1D") and **MUST NOT** use "TNN" as a shorthand.

---

## 3. Lab1-Specific Naming

| Old (Banned) | New (Required) |
|---|---|
| "Topological TNN" | "ResConv1D (residual baseline)" |
| "Hamiltonian Energy Drift" | "L2-Norm Drift" (with dissipative caveat) |
| "Live Sonic Horizon Wave Simulator" | "Live Advection-Diffusion Kinematic Proxy" |
| "SONIC HORIZON (Fr=1)" | "Fr=1 PROXY LINE (eff_vel sign change)" |
| Torres et al. (2017) citation | REMOVED — requires azimuthal rotation |

---

## 4. When "Poly-Algebraic" May Appear

- ✅ In `specs/Foundations of Poly-Algebraic Calculus.md` (the theoretical paper)
- ✅ In `vhpu_engine/src/main.rs` comments (describing the concept being emulated)
- ✅ In research paper `.tex` files (formal academic context)
- ❌ In dashboard UI labels (misleads about what the code executes)
- ❌ In Python script variable names or class names (unless implementing the formal calculus)

---

## 5. Compliance Check

Before any commit, run:
```bash
# Find potential violations (exclude specs/, docs/, and this file)
grep -rn --include='*.py' --include='*.js' --include='*.html' \
  -e 'Topological TNN' -e 'Poly-Algebraic' \
  --exclude-dir=specs --exclude-dir=docs --exclude-dir=lib \
  scripts/ dashboard/ tnn/ web/
```

Any match in runtime code (not specs/docs) must be justified or renamed.
