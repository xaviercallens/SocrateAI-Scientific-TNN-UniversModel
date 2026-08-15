# CERTIFIED SCIENTIFIC AUDIT REPORT & REPRODUCIBLE PROTOCOL
## LAB-1: Analogue Gravity Proxy — 1D Advection-Diffusion Toy Model

**Audit Version:** v2.1  
**Date:** 2026-08-15  
**Certificate SHA-256:** `053c3b694c03fefe35a53b722a20f7cc1759f1da7a1b5a8547fbcbfad4a15638`  
**Dataset SHA-256:** `aad33202a91404785969629072891cd3a36f4b9068b6b47e2c1ef1477c5bb9d9`  
**Verification:** 20/20 machine checks pass

---

## What This Benchmark Is — and Is Not

The simulator integrates the **1D scalar advection-diffusion equation**:

$$\frac{\partial u}{\partial t} = -v_{\text{eff}}(x) \cdot \frac{\partial u}{\partial x} + \varepsilon \cdot \frac{\partial^2 u}{\partial x^2}, \quad v_{\text{eff}}(x) = U(x) - c(x)$$

where $v_{\text{eff}}$ is computed once and **frozen**. This is a **PDE-interpolation proxy** — not the true two-field (height + velocity) shallow-water system of Weinfurtner (2011). Both networks learn to imitate a deterministic finite-difference scheme. Neither "sees" $\text{Fr}=1$ as a special object.

---

## Corrections Applied (P1–P7)

| # | Problem (Original) | Correction | Verified |
|---|---|---|---|
| **P1** | "Live Sonic Horizon Simulator" / "analogous to Hawking white hole" | Renamed "1D Advection-Diffusion Proxy". Honesty note in all files. | ✅ |
| **P2** | Networks imitate numerical scheme, not horizon physics | Added: *"PDE-interpolation test, not a horizon-physics test"* everywhere | ✅ |
| **P3** | CNN 62k vs TNN 167k params (2.7× bias) | Equalized: CNN **112,489** vs ResConv1D **94,609** (1.19×) | ✅ |
| **P4** | "Hamiltonian drift" misuses symplectic vocabulary | Renamed to **L2-Norm Drift** everywhere. Dissipative caveat added. | ✅ |
| **P5** | Torres 2017 cited without rotation in model | Citation **removed**. Reason documented. | ✅ |
| **P6** | "Topological TNN", "Poly-Algebraic Calculus" in code | Renamed to **ResConv1D (residual baseline)** everywhere. | ✅ |
| **P7** | Governance not audited | Dashboard branding corrected. Formal audit pending. | ⚠️ |

---

## Benchmark Results (Equal Parameter Budget)

### With Horizon (Fr ≥ 1)

| Architecture | Parameters | Test MSE | L2-Norm Drift | Time (s) |
|---|---|---|---|---|
| Baseline CNN 1D | 112,489 | $6.140 \times 10^{-5}$ | 1.219% | 552 |
| **ResConv1D (residual)** | 94,609 | $1.604 \times 10^{-5}$ | **0.083%** | 313 |
| **Ratio** | 1.19× | **3.83× better** | **14.7× better** | — |

### Negative Control (Fr < 1, No Horizon)

| Architecture | Test MSE | Ratio |
|---|---|---|
| Baseline CNN 1D | $2.220 \times 10^{-4}$ | — |
| ResConv1D (residual) | $2.083 \times 10^{-5}$ | **10.66× better** |

**Critical finding:** The ResConv1D advantage is **2.78× larger without a horizon** (10.66×) than with one (3.83×). `validates_horizon_specificity: false`. The gap measures **generic smooth-field interpolation capacity**, not horizon-specific physics.

---

## Literature Sources

1. **Unruh (1981)** — PRL 46, 1351 — Kinematic acoustic horizon analogy
2. **Rousseaux et al. (2008)** — NJP 10, 053015 — Mode conversion at horizon
3. **Weinfurtner et al. (2011)** — PRL 106, 021302 — Source of $h_0=0.24\,\text{m}$, $U_0=0.668\,\text{m/s}$
4. ~~Torres et al. (2017)~~ — **Removed** — Requires rotation, model has none
5. **Crowther et al. (2019)** — Synthese 198(10) — Mathematical analogy ≠ physical identity

---

## Reproducible Execution

```bash
python3 scripts/certified_audit_lab1.py
python3 -m json.tool certs/audit_certificate_lab1.json
python3 -m http.server 8080 --directory web/
```
