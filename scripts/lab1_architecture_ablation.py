"""
================================================================================
LAB-1 ARCHITECTURE ABLATION STUDY
================================================================================
PURPOSE:
  The certified audit (v3.0) shows ResConv1D outperforms Baseline CNN by 3.83×
  in MSE on the 1D advection-diffusion proxy. The audit document acknowledges
  this as an "ablation hypothesis" but does not isolate the source of the gain.

  This script performs a controlled ablation to determine which of the three
  architectural differences drives the performance gap:

    A. RESIDUAL SKIP CONNECTIONS  — gradient flow through identity shortcut
    B. GROUPNORM (vs BatchNorm)   — cross-sample normalization for varied physics
    C. GELU (vs ReLU)             — smooth nonlinearity with negative saturation

  We test 8 model variants (2³ ablation grid) to isolate each contribution.

EXPECTED OUTCOME:
  - GroupNorm is hypothesized to contribute most, because it normalizes within a
    sample rather than across batches, making it robust to the randomized physics
    (each sample has a different U0/bump_height → different scale).
  - Residuals are secondary — they help gradient flow in deeper networks.
  - GELU has a small but consistent positive effect.

  If GroupNorm is NOT the dominant driver, the honest conclusion is that the
  ResConv1D advantage is primarily a depth/capacity effect, not a normalization one.

HONESTY STATEMENT:
  Results are purely empirical on the advection-diffusion proxy.
  This is NOT a claim about any topological structure.
  This is NOT a claim about horizon-specific physics.
================================================================================
"""

import time
import json
import datetime
import hashlib
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
torch.manual_seed(SEED)

RESOLUTION  = 256
HIDDEN      = 48        # Same hidden size for all ablation variants
NUM_TRAIN   = 2000
NUM_TEST    = 500
EPOCHS      = 20        # Enough epochs for fair comparison at HIDDEN=48
BATCH_SIZE  = 64
LR          = 1e-3


# ── Dataset generation (same as certified_audit_lab1.py v3.0) ────────────────

def simulate_kinematic_proxy(num_samples=2500, resolution=256, force_no_horizon=False):
    """
    1D advection-diffusion proxy — 2-channel input, per-sample randomized physics.
    Identical to certified_audit_lab1.py v3.0 simulate_kinematic_proxy().
    """
    x = torch.linspace(-2.0, 2.0, resolution)
    dx = x[1] - x[0]
    dt = 0.002
    g = 9.81
    h0 = 0.24

    inputs  = torch.zeros(num_samples, 2, resolution)
    targets = torch.zeros(num_samples, 1, resolution)

    for i in range(num_samples):
        if force_no_horizon:
            bump_height = 0.02 + 0.02 * torch.rand(1).item()
            U0 = 0.3 + 0.1 * torch.rand(1).item()
        else:
            bump_height = 0.16 + 0.025 * torch.randn(1).item()
            bump_height = max(0.08, min(0.22, bump_height))
            U0 = 0.668 + 0.06 * torch.randn(1).item()
            U0 = max(0.45, min(0.90, U0))

        h = h0 - bump_height * torch.exp(-(x**2) / 0.15)
        h = torch.clamp(h, min=0.02)
        U = (U0 * h0) / h
        c_speed = torch.sqrt(g * h)
        eff_vel = U - c_speed

        x0 = 0.8 + 0.8 * torch.rand(1).item()
        w  = 0.08 + 0.04 * torch.rand(1).item()
        u_t = torch.exp(-((x - x0) / w)**2)

        u_next = u_t.clone()
        for _ in range(15):
            du_dx   = torch.zeros_like(u_next)
            d2u_dx2 = torch.zeros_like(u_next)
            du_dx[1:-1]   = (u_next[2:] - u_next[:-2]) / (2 * dx)
            d2u_dx2[1:-1] = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_next = u_next + dt * (-eff_vel * du_dx + 0.001 * d2u_dx2)

        mean_u = u_t.mean()
        std_u  = u_t.std() + 1e-6
        inputs[i, 0, :]  = (u_t    - mean_u) / std_u
        inputs[i, 1, :]  = eff_vel
        targets[i, 0, :] = (u_next - mean_u) / std_u

    return inputs, targets


# ── Ablation model factory ────────────────────────────────────────────────────

class AblationConv1D(nn.Module):
    """
    Parametric 1D convolutional model for ablation study.

    Ablation axes:
      use_residual  : True → add residual skip connections
      use_groupnorm : True → GroupNorm, False → BatchNorm
      use_gelu      : True → GELU activation, False → ReLU

    All other hyperparameters held constant:
      in_ch=2, out_ch=1, hidden=48, layers=4, kernel=5
    """
    def __init__(self, in_ch=2, out_ch=1, hidden=HIDDEN, layers=4,
                 use_residual=True, use_groupnorm=True, use_gelu=True):
        super().__init__()
        self.use_residual = use_residual

        activation = nn.GELU if use_gelu else nn.ReLU
        norm_class = (lambda c: nn.GroupNorm(min(8, c), c)) if use_groupnorm \
                     else (lambda c: nn.BatchNorm1d(c))

        self.lift = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            if use_residual:
                self.blocks.append(nn.ModuleDict({
                    "conv1": nn.Conv1d(hidden, hidden, 5, padding=2),
                    "norm1": norm_class(hidden),
                    "act":   activation(),
                    "conv2": nn.Conv1d(hidden, hidden, 5, padding=2),
                    "norm2": norm_class(hidden),
                }))
            else:
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    norm_class(hidden), activation(),
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    norm_class(hidden), activation(),
                ))

        self.proj = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 1), activation(),
            nn.Conv1d(hidden // 2, out_ch, 1)
        )

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            if self.use_residual:
                res = h
                h = block["act"](block["norm1"](block["conv1"](h)))
                h = block["norm2"](block["conv2"](h))
                h = h + res   # residual skip
            else:
                h = block(h)
        return self.proj(h)


# ── Train + eval ─────────────────────────────────────────────────────────────

def run_variant(config, x_tr, y_tr, x_te, y_te):
    name, use_res, use_gn, use_gelu = config
    model = AblationConv1D(use_residual=use_res, use_groupnorm=use_gn, use_gelu=use_gelu)
    params = sum(p.numel() for p in model.parameters())
    opt    = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(x_tr[idx]), y_tr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        mse_te = loss_fn(model(x_te), y_te).item()

    return {"name": name, "params": params, "mse": mse_te, "time_s": round(elapsed, 1),
            "use_residual": use_res, "use_groupnorm": use_gn, "use_gelu": use_gelu}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("  LAB-1 ARCHITECTURE ABLATION STUDY (2³ = 8 variants)")
    print("  Isolating: Residual | GroupNorm | GELU contributions to ResConv1D gap")
    print("=" * 80)

    print("\nGenerating dataset (2-channel, randomized physics)...")
    x_all, y_all = simulate_kinematic_proxy(
        num_samples=NUM_TRAIN + NUM_TEST, resolution=RESOLUTION
    )
    data_hash = hashlib.sha256(x_all.numpy().tobytes()).hexdigest()
    print(f"  Dataset SHA256: {data_hash[:32]}...")

    x_tr, y_tr = x_all[:NUM_TRAIN], y_all[:NUM_TRAIN]
    x_te, y_te = x_all[NUM_TRAIN:], y_all[NUM_TRAIN:]

    # All 8 combinations (R, G, A) = (residual, groupnorm, gelu)
    variants = []
    for res, gn, gelu in itertools.product([False, True], repeat=3):
        parts = [
            "Res" if res else "NoRes",
            "GN"  if gn  else "BN",
            "GELU" if gelu else "ReLU"
        ]
        variants.append(("-".join(parts), res, gn, gelu))

    results = []
    for cfg in variants:
        print(f"\n  Training: {cfg[0]}...")
        r = run_variant(cfg, x_tr, y_tr, x_te, y_te)
        results.append(r)
        print(f"    MSE={r['mse']:.4e}  params={r['params']:,}  time={r['time_s']}s")

    # Sort by MSE (best first)
    results.sort(key=lambda x: x["mse"])

    print("\n" + "=" * 80)
    print("  ABLATION RESULTS (sorted by MSE — lower is better)")
    print("  " + "-" * 76)
    print(f"  {'Variant':<30} {'MSE':>10} {'Params':>8} {'Rank':>5}")
    print("  " + "-" * 76)
    worst_mse = results[-1]["mse"]
    best_mse  = results[0]["mse"]
    for rank, r in enumerate(results, 1):
        speedup = worst_mse / (r["mse"] + 1e-12)
        print(f"  {r['name']:<30} {r['mse']:>10.4e} {r['params']:>8,}  #{rank}  ({speedup:.2f}x vs worst)")

    # Marginal contribution analysis
    # Average gain from each axis across all other combinations
    axes = {"Residual": "use_residual", "GroupNorm": "use_groupnorm", "GELU": "use_gelu"}
    print("\n  MARGINAL CONTRIBUTION (average MSE improvement when axis is ON vs OFF):")
    for axis_name, axis_key in axes.items():
        on_mses  = [r["mse"] for r in results if     r[axis_key]]
        off_mses = [r["mse"] for r in results if not r[axis_key]]
        avg_on  = sum(on_mses)  / len(on_mses)
        avg_off = sum(off_mses) / len(off_mses)
        improvement = (avg_off - avg_on) / avg_off * 100
        verdict = "↑ HELPS" if avg_on < avg_off else "↓ HURTS"
        print(f"    {axis_name:<12}: avg_off={avg_off:.3e}  avg_on={avg_on:.3e}  "
              f"→ {improvement:+.1f}% MSE change  {verdict}")

    # Final honest interpretation
    best = results[0]
    print("\n  HONEST INTERPRETATION:")
    print(f"    Best variant: {best['name']}  (MSE={best['mse']:.4e})")
    print(f"    Worst variant: {results[-1]['name']}  (MSE={results[-1]['mse']:.4e})")
    print(f"    Total architecture effect: {best_mse/worst_mse:.3f}x → "
          f"{(1 - best_mse/worst_mse)*100:.1f}% MSE reduction from architectural choices")
    print("\n    NOTE: All differences are on a PDE-interpolation benchmark, not on")
    print("    horizon-specific physics. The 'best architecture' here is the")
    print("    best LOCAL SMOOTH-FIELD INTERPOLATOR, not the best horizon detector.")

    # Write results
    import os
    os.makedirs("certs", exist_ok=True)
    output = {
        "script": "lab1_architecture_ablation.py",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "dataset_sha256": data_hash,
        "honesty_statement": (
            "Ablation performed on 1D advection-diffusion proxy (NOT horizon physics). "
            "Results quantify architecture-driven interpolation capacity, not physical discovery."
        ),
        "num_train": NUM_TRAIN,
        "num_test": NUM_TEST,
        "epochs": EPOCHS,
        "hidden": HIDDEN,
        "resolution": RESOLUTION,
        "results": results,
        "best_variant": best["name"],
        "mse_range_factor": round(worst_mse / best_mse, 3)
    }
    with open("certs/ablation_certificate_lab1.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Certificate written to: certs/ablation_certificate_lab1.json")
    print("=" * 80)
