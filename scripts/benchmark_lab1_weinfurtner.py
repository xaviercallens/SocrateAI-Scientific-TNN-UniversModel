"""
=================================================================
  LAB-1 BENCHMARK: Analogue Gravity Flume
  Literature Parameters (Weinfurtner 2011 / Unruh 1981)
=================================================================
NAMING POLICY (Rule RES-1 / N-1):
  ResConv1D = Spectral-Residual Baseline (GroupNorm + GELU + residual skip).
  NOT "Topological". No topological structure exists in this code.
  Reserve "Topological" exclusively for architectures with verified TDA.

HONESTY: This is a 1D scalar advection-diffusion proxy with STATIC eff_vel.
The coupled (h, v) Saint-Venant equations are NOT solved.
For per-sample randomized physics, use: certified_audit_lab1.py v3.0
"""
import time, json, math, datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ─────────────────── DATA GENERATION ───────────────────
def generate_weinfurtner_flume(num_samples=2000, resolution=256, seed=42):
    """1D advection-diffusion proxy. eff_vel is STATIC (frozen across samples)."""
    torch.manual_seed(seed)
    x  = torch.linspace(-2.0, 2.0, resolution)
    dx = x[1] - x[0]
    dt = 0.005
    g  = 9.81
    h0 = 0.24
    bump_height = 0.18
    h  = h0 - bump_height * torch.exp(-x**2 / 0.2)
    U0 = 0.668
    U  = (U0 * h0) / h
    c  = torch.sqrt(g * h)
    eff_vel = U - c

    u = torch.zeros(num_samples, resolution)
    for i in range(num_samples):
        x0    = 1.0 + 1.0 * torch.rand(1).item()
        width = 0.1 + 0.1 * torch.rand(1).item()
        u[i]  = torch.exp(-((x - x0) / width)**2)

    u_t = u.clone()
    for _ in range(20):
        du_dx   = torch.zeros_like(u_t)
        d2u_dx2 = torch.zeros_like(u_t)
        du_dx[:, 1:-1]   = (u_t[:, 2:] - u_t[:, :-2]) / (2 * dx)
        d2u_dx2[:, 1:-1] = (u_t[:, 2:] - 2*u_t[:, 1:-1] + u_t[:, :-2]) / (dx**2)
        du_dt = -eff_vel.unsqueeze(0) * du_dx + 0.05 * d2u_dx2
        u_t   = u_t + dt * du_dt

    return u.unsqueeze(1), u_t.unsqueeze(1)

# ─────────────────── MODELS ───────────────────
class BaselineCNN1D(nn.Module):
    """Naive local-convolution baseline. BatchNorm + ReLU."""
    def __init__(self, in_ch=1, out_ch=1, hidden=64, layers=4):
        super().__init__()
        mods = [nn.Conv1d(in_ch, hidden, 5, padding=2), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU()]
        mods.append(nn.Conv1d(hidden, out_ch, 1))
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class ResConv1D(nn.Module):
    """
    Spectral-Residual Baseline. GroupNorm + GELU + residual skip connections.
    NOT topological. Advantage over CNN is hypothesized from:
    (1) residual skip → smoother gradients
    (2) GroupNorm → stable normalization across spatial positions
    (3) GELU → smoother activation landscape
    These are ablation hypotheses, not proven structural claims.
    """
    def __init__(self, in_ch=1, out_ch=1, hidden=64, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GroupNorm(8, hidden), nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GroupNorm(8, hidden),
            ) for _ in range(layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(hidden, hidden//2, 1), nn.GELU(),
            nn.Conv1d(hidden//2, out_ch, 1)
        )
    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = h + block(h)
        return self.proj(h)

# ─────────────────── TRAINING ───────────────────
def train_loop(model, x_tr, y_tr, x_te, y_te, epochs=15, lr=1e-3, bs=64):
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n       = len(x_tr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return loss_fn(model(x_te), y_te).item()

# ─────────────────── RUN ───────────────────
if __name__ == "__main__":
    print("="*75)
    print("  LAB-1: ANALOGUE GRAVITY FLUME (Weinfurtner 2011 / Rousseaux 2008)")
    print("  Spectral-Residual Baseline vs Naive CNN (LEGACY — static eff_vel)")
    print("  For rigorous benchmark: use certified_audit_lab1.py v3.0")
    print("="*75)

    t0 = time.time()
    u_in, u_out = generate_weinfurtner_flume(2000)
    print(f"Dataset generated in {time.time()-t0:.2f}s | h0=0.24m | U0=0.668m/s")

    xtr, ytr = u_in[:1600], u_out[:1600]
    xte, yte = u_in[1600:], u_out[1600:]

    b_params = sum(p.numel() for p in BaselineCNN1D(hidden=64).parameters())
    r_params = sum(p.numel() for p in ResConv1D(hidden=64).parameters())
    ratio_p  = max(b_params, r_params) / min(b_params, r_params)
    print(f"  Params — CNN: {b_params:,} | ResConv1D: {r_params:,} | Ratio: {ratio_p:.2f}x")
    if ratio_p > 1.2:
        print("  ⚠️  PARAM BUDGET WARNING: ratio > 1.2x — interpret MSE gap with caution")

    print("\nTraining Baseline CNN...")
    b_model = BaselineCNN1D(hidden=64, layers=4)
    b_mse   = train_loop(b_model, xtr, ytr, xte, yte, epochs=20)

    print("Training ResConv1D (Spectral-Residual Baseline)...")
    r_model = ResConv1D(hidden=64, layers=4)
    r_mse   = train_loop(r_model, xtr, ytr, xte, yte, epochs=20)

    ratio = b_mse / (r_mse + 1e-12)
    print("\n" + "="*75)
    print(f"  Baseline CNN MSE:  {b_mse:.4e}")
    print(f"  ResConv1D MSE:     {r_mse:.4e}  ({ratio:.1f}x better)")
    print(f"  NOTE: Gap measures PDE-interpolation on STATIC eff_vel (P2 bias).")
    print(f"  certified_audit_lab1.py v3.0 corrects this with randomized physics.")
    print("="*75)
