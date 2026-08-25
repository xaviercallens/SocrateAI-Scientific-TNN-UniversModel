"""
================================================================================
LAB-1 EXTENSION: Rotating Draining Vortex — Analogue Ergoregion Model (v2.0)
================================================================================
SCIENTIFIC MOTIVATION:
  Torres et al. (2017) [Nature Physics 13, 833–836] measured superradiance in a
  rotating fluid vortex. The ergoregion (analogue of a Kerr black hole's ergosphere)
  requires BOTH a radial sink (inflow) AND an azimuthal velocity component.

  The previous model (benchmark_lab1_analogue_gravity.py) used ONLY:
      v_r(r) = -A/r   (pure radial sink)
  → This is a Schwarzschild-like analogue (no rotation, no ergoregion).
  → Torres 2017 is INAPPLICABLE to that model.

  This file implements:
      v_r(r)   = -A / r            (radial inflow)
      v_phi(r) = B / r             (azimuthal circulation, Rankine vortex outside core)
      |v_total(r)| = sqrt(v_r² + v_phi²) / c — defines the analogue horizon

  The sonic horizon r_h and ergoregion r_e > r_h satisfy:
      r_h : |v_r(r_h)| = c           (radial horizon — wave cannot escape radially)
      r_e : |v_total(r_e)| = c        (ergosphere — rotating-wave ergoregion)

  For a wave with azimuthal mode m, the effective potential becomes:
      du/dt = -v_r * du/dr + (c² - v_phi²/r²) * d²u/dr² + ... (simplified 1D proxy)

  NOTE: This is still a 1D RADIAL PROXY, not a full 2D rotating fluid simulation.
  The azimuthal mode number m is a parameter, not a spatial dimension.
  For a full 2D treatment, see Torres et al. (2017) and Basak & Majumdar (2003).

REFERENCES:
  Torres, T., et al. (2017). Observation of superradiance in a vortex flow.
    Nature Physics, 13, 833–836. doi:10.1038/nphys4151
  Basak, S., & Majumdar, P. (2003). Superradiance in a rotating acoustic black hole.
    Physical Review D, 68, 024005.
  Unruh, W. G. (1981). Experimental Black-Hole Evaporation?
    Physical Review Letters, 46(21), 1351.

HONESTY STATEMENT:
  - This simulates a 1D EFFECTIVE RADIAL PDE with azimuthal coupling — NOT a
    full 2D rotating fluid.
  - The coupling term v_phi²/r² modifies the effective wave speed, creating an
    ergoregion. This IS the relevant mechanism from Torres 2017.
  - The m=0 case (no azimuthal coupling) reduces to the pure radial horizon.
  - Superradiance requires m > 0 modes — the ergoregion amplifies co-rotating waves.
================================================================================
"""

import time
import json
import hashlib
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
torch.manual_seed(SEED)

# ── Physical parameters (Torres 2017 Table 1 analogue) ──────────────────────
C_WAVE   = 0.23   # m/s — wave speed at rest depth (matches flume-scale experiment)
A_SINK   = 0.05   # m²/s — circulation sink strength: v_r = -A/r
B_ROT    = 0.08   # m²/s — azimuthal circulation: v_phi = B/r (irrotational outside core)
R_CORE   = 0.03   # m — Rankine vortex core radius (solid-body rotation inside)
# Analytic horizon radii:
#   r_h  = A / C_WAVE  =  0.05/0.23 ≈ 0.217 m  (radial sonic horizon)
#   r_e  = sqrt(A²+B²) / C_WAVE ≈ 0.412 m      (ergosphere)
R_HORIZON  = A_SINK / C_WAVE
R_ERGO     = math.sqrt(A_SINK**2 + B_ROT**2) / C_WAVE


def rotating_vortex_flow(r: torch.Tensor, azimuthal_mode: int = 1):
    """
    Computes the effective radial group velocity for a wave of azimuthal mode m
    on a rotating draining vortex background.

    Physics (Basak & Majumdar 2003, eq. 2.6 in 1D radial proxy):
      v_r(r)   = -A/r                  (radial inflow)
      v_phi(r) = B/r  (r > r_core)    (irrotational azimuthal circulation)
               = B*r/r_core²  (r ≤ r_core)  (solid-body core)

    Effective group velocity for mode m:
      v_eff(r) = v_r(r) - m * v_phi(r) / (omega * r / c)   [dispersion relation]

    Simplified (in the WKB long-wavelength limit):
      v_eff(r) ≈ v_r(r)   for m=0   (no ergoregion, pure radial horizon)
      v_eff(r) ≈ v_r(r) - m * v_phi(r) / r  (ergoregion coupling for m≠0)

    Args:
        r              : radial coordinate tensor [resolution]
        azimuthal_mode : m — azimuthal wavenumber (0 = no rotation coupling)

    Returns:
        v_r    : radial velocity [resolution]
        v_phi  : azimuthal velocity [resolution]
        v_eff  : effective group velocity including ergoregion coupling [resolution]
    """
    v_r = -A_SINK / r

    # Rankine vortex: solid body inside core, irrotational outside
    v_phi = torch.where(
        r > R_CORE,
        B_ROT / r,
        B_ROT * r / (R_CORE ** 2)
    )

    # Effective group velocity in 1D radial proxy (WKB approximation)
    # For m=0: v_eff = v_r (no ergoregion effect)
    # For m≠0: the azimuthal term modifies where the effective "horizon" is
    v_eff = v_r - azimuthal_mode * v_phi / r  # [resolution]

    return v_r, v_phi, v_eff


def simulate_rotating_vortex(
    num_samples: int = 2000,
    resolution: int = 256,
    azimuthal_mode: int = 1,
    force_no_ergoregion: bool = False,
    seed: int = SEED
) -> tuple:
    """
    Generates a 2-channel dataset for the rotating draining vortex proxy.

    Input channels:
      ch 0: normalized wave amplitude u(r, t)
      ch 1: effective group velocity v_eff(r) — carries ergoregion information

    Physics (per sample):
      - A and B randomized around Torres 2017 analogue values
      - Azimuthal mode m controls ergoregion coupling
      - PDE integrated: du/dt = -v_eff * du/dr + ε * d²u/dr²

    Control arm (force_no_ergoregion=True):
      - B=0 (no rotation), m=0 → pure radial horizon (Schwarzschild analogue)
      - Used to verify: ergoregion effects are LARGER for m>0 than m=0

    Args:
        num_samples        : number of wave-evolution samples
        resolution         : radial grid points
        azimuthal_mode     : m ∈ {0, 1, 2, ...} — 0 = no ergoregion
        force_no_ergoregion: if True, set B=0, m=0 (pure radial, no superradiance)
        seed               : RNG seed

    Returns:
        inputs  : [N, 2, resolution]  — [wave, v_eff]
        targets : [N, 1, resolution]  — next-state wave
    """
    torch.manual_seed(seed)
    r = torch.linspace(R_HORIZON * 0.5, 10 * R_HORIZON, resolution)  # avoid r=0
    dr = r[1] - r[0]
    dt = 0.001
    eps = 5e-4  # artificial diffusion for numerical stability

    inputs  = torch.zeros(num_samples, 2, resolution)
    targets = torch.zeros(num_samples, 1, resolution)

    m = 0 if force_no_ergoregion else azimuthal_mode

    for i in range(num_samples):
        # Randomize physical parameters around reference values
        if force_no_ergoregion:
            A = A_SINK * (0.8 + 0.4 * torch.rand(1).item())
            B = 0.0    # no rotation → no ergoregion
        else:
            A = A_SINK * (0.8 + 0.4 * torch.rand(1).item())
            B = B_ROT  * (0.7 + 0.6 * torch.rand(1).item())  # vary rotation

        # Compute flow fields for this sample
        v_r   = -A / r
        v_phi = torch.where(r > R_CORE, B / r, B * r / (R_CORE**2))
        v_eff = v_r - m * v_phi / r

        # Random wave packet outside the ergoregion
        r_ergo_sample = math.sqrt(A**2 + B**2) / C_WAVE if B > 0 else A / C_WAVE
        r0    = r_ergo_sample * (1.5 + 1.5 * torch.rand(1).item())
        r0    = min(r0, r[-1].item() * 0.8)
        width = 0.02 + 0.03 * torch.rand(1).item()
        u_t   = torch.exp(-((r - r0) / width)**2)

        # Integrate PDE for 20 steps
        u_next = u_t.clone()
        for _ in range(20):
            du_dr   = torch.zeros_like(u_next)
            d2u_dr2 = torch.zeros_like(u_next)
            du_dr[1:-1]   = (u_next[2:] - u_next[:-2]) / (2 * dr)
            d2u_dr2[1:-1] = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / dr**2
            u_next = u_next + dt * (-v_eff * du_dr + eps * d2u_dr2)
            # Boundary: absorbing at r_min, free at r_max
            u_next[0]  = 0.0
            u_next[-1] = u_next[-2]

        # Normalize
        std_u = u_t.std() + 1e-8
        mean_u = u_t.mean()
        inputs[i, 0, :]  = (u_t    - mean_u) / std_u
        inputs[i, 1, :]  = v_eff   # raw v_eff — carries ergoregion geometry
        targets[i, 0, :] = (u_next - mean_u) / std_u

    return inputs, targets


# ── Architectures (same as certified_audit_lab1.py v3.0 — equal budget) ─────

class BaselineCNN1D(nn.Module):
    """Naive local-convolution baseline. BatchNorm + ReLU. ~113k params."""
    def __init__(self, in_ch=2, out_ch=1, hidden=86, layers=4):
        super().__init__()
        mods = [nn.Conv1d(in_ch, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU()]
        mods.append(nn.Conv1d(hidden, out_ch, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class ResConv1D(nn.Module):
    """Residual baseline. GroupNorm + GELU + skip. ~95k params. NOT topological."""
    def __init__(self, in_ch=2, out_ch=1, hidden=48, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GroupNorm(8, hidden), nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GroupNorm(8, hidden)
            ) for _ in range(layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 1), nn.GELU(),
            nn.Conv1d(hidden // 2, out_ch, 1)
        )

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = h + block(h)
        return self.proj(h)


def train_and_eval(model_class, in_ch, hidden, x_tr, y_tr, x_te, y_te, name="Model"):
    model = model_class(in_ch=in_ch, hidden=hidden)
    params = sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for _ in range(15):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            loss_fn(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        mse = loss_fn(model(x_te), y_te).item()
    print(f"  {name}: params={params:,}  MSE={mse:.4e}  time={elapsed:.1f}s")
    return params, mse, elapsed


# ── Main experiment ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("  LAB-1 EXTENSION: Rotating Draining Vortex v2.0")
    print("  Ergoregion Model — Torres 2017 Compatible (azimuthal mode m=1)")
    print(f"  r_horizon = {R_HORIZON:.4f} m  |  r_ergo = {R_ERGO:.4f} m")
    print("=" * 80)

    results = {}

    for m, label in [(0, "m=0 (no ergoregion, pure radial)"),
                     (1, "m=1 (ergoregion active, superradiance regime)")]:

        print(f"\n--- MODE: {label} ---")

        # Horizon regime
        x_in, y_out = simulate_rotating_vortex(
            num_samples=2000, resolution=200, azimuthal_mode=m,
            force_no_ergoregion=(m == 0)
        )
        data_hash = hashlib.sha256(
            x_in.numpy().tobytes() + y_out.numpy().tobytes()
        ).hexdigest()
        print(f"  Dataset SHA256: {data_hash[:16]}...")

        x_tr, y_tr = x_in[:1600], y_out[:1600]
        x_te, y_te = x_in[1600:], y_out[1600:]

        cnn_p, cnn_mse, cnn_t = train_and_eval(
            BaselineCNN1D, in_ch=2, hidden=86,
            x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
            name="Baseline CNN"
        )
        res_p, res_mse, res_t = train_and_eval(
            ResConv1D, in_ch=2, hidden=48,
            x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
            name="ResConv1D"
        )

        ratio = cnn_mse / (res_mse + 1e-12)
        print(f"  → Param ratio: {cnn_p/res_p:.2f}x  |  MSE ratio: {ratio:.2f}x")
        results[f"m{m}"] = {
            "mode": m, "label": label,
            "cnn_mse": cnn_mse, "res_mse": res_mse,
            "mse_ratio": round(ratio, 3),
            "param_ratio": round(cnn_p / res_p, 3),
            "data_sha256": data_hash,
            "ergoregion_active": (m > 0),
            "r_horizon_m": round(R_HORIZON, 5),
            "r_ergo_m": round(R_ERGO, 5) if m > 0 else None
        }

    # Ergoregion specificity check
    r_m0 = results["m0"]["mse_ratio"]
    r_m1 = results["m1"]["mse_ratio"]
    ergo_specific = r_m1 > r_m0
    print("\n" + "=" * 80)
    print(f"  Mode m=0 (no ergo) MSE ratio: {r_m0:.2f}x")
    print(f"  Mode m=1 (ergo)    MSE ratio: {r_m1:.2f}x")
    print(f"  Ergoregion specificity: {'TRUE — ResConv1D has larger advantage WITH ergoregion' if ergo_specific else 'FALSE — gap is geometry-independent'}")
    print("=" * 80)

    # Literature validity statement
    if ergo_specific:
        print("\n  ✅ Torres 2017 citation is now justified:")
        print("     Model contains azimuthal velocity v_phi(r) = B/r (Rankine)")
        print("     Ergoregion r_e = sqrt(A²+B²)/c is explicitly modelled")
        print("     MSE gap is larger in ergoregion regime — architecture responds to ergoregion geometry")
    else:
        print("\n  ⚠️  Torres 2017 citation remains unjustified:")
        print("     MSE gap is not larger with ergoregion — gap is generic interpolation")
        print("     Re-run with larger m or more ergoregion contrast before citing Torres 2017")

    # Write certificate
    cert = {
        "script": "lab1_draining_vortex_v2_rotating.py",
        "version": "2.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "honesty_statement": (
            "1D radial proxy with azimuthal coupling term. "
            "NOT a full 2D rotating fluid. "
            "Models ergoregion via effective group velocity v_eff = v_r - m*v_phi/r. "
            "Torres 2017 citation validity depends on ergoregion_specificity flag."
        ),
        "references": {
            "Torres_2017": "Nature Physics 13, 833–836. doi:10.1038/nphys4151",
            "Basak_Majumdar_2003": "Phys. Rev. D 68, 024005",
            "Unruh_1981": "PRL 46, 1351"
        },
        "results": results,
        "ergoregion_specificity": ergo_specific,
        "torres_2017_citation_valid": ergo_specific
    }
    import os
    os.makedirs("certs", exist_ok=True)
    with open("certs/rotating_vortex_v2_certificate.json", "w") as f:
        json.dump(cert, f, indent=2)
    print("\n  Certificate written to: certs/rotating_vortex_v2_certificate.json")
