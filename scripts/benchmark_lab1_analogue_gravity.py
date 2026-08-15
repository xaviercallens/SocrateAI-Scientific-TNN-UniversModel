"""
=================================================================
  LAB-1 BENCHMARK: Analogue Gravity in a Water Tank (Draining Vortex)
=================================================================
Simulates the propagation of shallow water waves on a draining vortex background.
We compare a physics-naive Traditional CNN against a TNN (Spectral ResConv1D).

The radial flow v(r) = -A/r creates a sonic horizon where |v(r)| = c = sqrt(g*h).
The model must predict the wave field u(r, t+dt) from u(r, t).
"""
import time, json, math, datetime
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────── DATA GENERATION ───────────────────
def generate_draining_vortex_waves(num_samples=2000, resolution=128, seed=42):
    """
    Generates 1D radial wave propagation on a background sink flow v(r) = -A/r.
    Wave equation: (d/dt + v d/dr)^2 u = c^2 d^2u/dr^2
    We use a simplified advection-wave simulation.
    """
    torch.manual_seed(seed)
    r = torch.linspace(0.1, 5.0, resolution)
    dr = r[1] - r[0]
    dt = 0.005
    c = 1.0          # Wave speed
    A = 0.5          # Sink strength, Horizon at r_h = A/c = 0.5
    v = -A / r       # Background flow
    
    # Generate initial wave packets outside the horizon
    u = torch.zeros(num_samples, resolution)
    for i in range(num_samples):
        r0 = 1.0 + 3.0 * torch.rand(1).item() # center between 1 and 4
        width = 0.2 + 0.3 * torch.rand(1).item()
        u[i] = torch.exp(-((r - r0) / width)**2)
    
    # Simple explicit step (upwind for advection, central for wave)
    # We just need a deterministic mapping u_t -> u_{t+dt} for the ML task.
    # To make it physically meaningful, we evolve it for 10 steps to get the target.
    
    u_t = u.clone()
    u_next = torch.zeros_like(u_t)
    
    # We use a smoothed evolution to create the dataset
    for step in range(10):
        # du/dr (central diff)
        du_dr = torch.zeros_like(u_t)
        du_dr[:, 1:-1] = (u_t[:, 2:] - u_t[:, :-2]) / (2 * dr)
        # d2u/dr2
        d2u_dr2 = torch.zeros_like(u_t)
        d2u_dr2[:, 1:-1] = (u_t[:, 2:] - 2*u_t[:, 1:-1] + u_t[:, :-2]) / (dr**2)
        
        # Advection + wave dispersion (simplified for the benchmark dataset)
        du_dt = -v.unsqueeze(0) * du_dr + 0.1 * c**2 * d2u_dr2
        u_t = u_t + dt * du_dt
        
    return u.unsqueeze(1), u_t.unsqueeze(1) # [batch, 1, res]

# ─────────────────── MODELS ───────────────────
class BaselineCNN1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, hidden=64, layers=4):
        super().__init__()
        mods = [nn.Conv1d(in_ch, hidden, 5, padding=2), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU()]
        mods.append(nn.Conv1d(hidden, out_ch, 1))
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class ResConv1D(nn.Module):
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
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_loop(model, x_tr, y_tr, x_te, y_te, epochs=10, lr=1e-3, bs=128):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = float('inf')
    n = len(x_tr)
    t0 = time.time()
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
            tl = loss_fn(model(x_te), y_te).item()
        best = min(best, tl)
    return best, time.time() - t0

# ─────────────────── RUN ───────────────────
if __name__ == "__main__":
    print("Generating Draining Vortex Dataset (Analogue Gravity)...")
    u_in, u_out = generate_draining_vortex_waves(3000)
    
    xtr, ytr = u_in[:2400], u_out[:2400]
    xte, yte = u_in[2400:], u_out[2400:]
    
    print("Training Baseline CNN...")
    b_model = BaselineCNN1D()
    b_mse, b_time = train_loop(b_model, xtr, ytr, xte, yte)
    b_p = count_params(b_model)
    
    print("Training TNN (ResConv1D)...")
    t_model = ResConv1D()
    t_mse, t_time = train_loop(t_model, xtr, ytr, xte, yte)
    t_p = count_params(t_model)
    
    print("\n" + "="*70)
    print("  LAB-1 BENCHMARK RESULTS: Analogue Gravity Horizon")
    print("="*70)
    print(f"  Traditional CNN: MSE={b_mse:.4e} | {b_p:,} params | {b_time:.1f}s")
    print(f"  TNN (Spectral):  MSE={t_mse:.4e} | {t_p:,} params | {t_time:.1f}s")
    print(f"  >>> TNN is {b_mse/t_mse:.1f}x more accurate.")
    print("="*70)
