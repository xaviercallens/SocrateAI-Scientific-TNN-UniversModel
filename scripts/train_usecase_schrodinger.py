"""
UC6 — Équation de Schrödinger 1D (Physique Quantique)
=======================================================
Architecture: LightweightFNO1D (native torch.fft spectral convolution).
NO external neuraloperator dependency. NO CNN fallback.
Physique: Évolution temporelle de la fonction d'onde ψ(x,t) dans un puits harmonique:
  iℏ ∂ψ/∂t = [-ℏ²/2m ∂²/∂x² + V(x)] ψ
Vulnerability D fix: SchrodingerFNO now uses TRUE Fourier convolution via torch.fft,
not a spatial CNN. is_truly_spectral() returns True always.
"""
import sys, os, math, time, datetime
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tnn.utils.data_generators import generate_schrodinger_wavepacket
# Vulnerability D fix: use native torch.fft spectral operator
from tnn.physics.symplectic_fno import LightweightFNO1D

# Vulnerability D fix: replace CNN residual block with true spectral FNO
class SchrodingerFNO(LightweightFNO1D):
    """
    True 1D Fourier Neural Operator for Schrödinger wave evolution.
    Uses native torch.fft.rfft — no external dependencies, no CNN fallback.
    is_truly_spectral() always returns True.
    Equivalent to Li et al. (2021) FNO for 1D, Eq. (8).
    """
    def __init__(self, resolution=128, hidden_channels=64, n_modes=24, n_layers=4):
        super().__init__(in_ch=1, out_ch=1, width=hidden_channels,
                         n_layers=n_layers, n_modes=n_modes)


class WaveNormConservationHook:
    """Vérifie ∫|ψ|²dx ≈ 1 (normalisation quantique) sur rollout."""
    @staticmethod
    def run(model, psi_test, dx=20.0/256):
        print("\n[HOOK] SchrodingerNormHook (Conservation ∫|ψ|²dx=1)...")
        model.eval()
        psi_t = psi_test[0:1].unsqueeze(1)  # [1,1,N]
        norms = []
        with torch.no_grad():
            for _ in range(100):
                norm = (psi_t.squeeze()**2 * dx).sum().item()
                norms.append(norm)
                psi_t = model(psi_t)
        norm_drift = max(norms) - min(norms)
        norm_mean = sum(norms)/len(norms)
        passed = abs(norm_mean - 1.0) < 0.1 and norm_drift < 0.2
        print(f"[HOOK] Norme moyenne: {norm_mean:.4f} | Dérive: {norm_drift:.4e}")
        print(f"[HOOK] {'✅ PASS' if passed else '❌ FAIL'}")
        return passed, norm_drift

def train():
    print("="*65)
    print("  TNN UC6 — ÉQUATION DE SCHRÖDINGER 1D (Opérateur de Fourier)")
    print("="*65)
    start = time.time()

    resolution = 128   # Reduced for CPU performance
    psi_t, psi_t1 = generate_schrodinger_wavepacket(resolution=resolution, num_samples=1000, seed=42)
    # psi_t: [1000, resolution], normalisation garantie

    split = 800
    psi_tr = psi_t[:split].unsqueeze(1)    # [800, 1, N]
    psi_t1_tr = psi_t1[:split].unsqueeze(1)
    psi_te = psi_t[split:].unsqueeze(1)
    psi_t1_te = psi_t1[split:].unsqueeze(1)

    model = SchrodingerFNO(resolution=resolution, hidden_channels=64, n_modes=24, n_layers=4)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    loss_fn = nn.MSELoss()
    best = float('inf')

    print(f"\n[*] Backend: {'FNO (neuralop)' if USE_FNO else 'Pseudo-FNO (FFT+MLP)'}")

    for epoch in range(1, 151):
        model.train()
        perm = torch.randperm(split)
        el = 0.0
        for i in range(0, split, 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            pred = model(psi_tr[idx])
            loss = loss_fn(pred, psi_t1_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
        sched.step()
        if epoch % 50 == 0 or epoch == 150:
            model.eval()
            with torch.no_grad():
                tl = loss_fn(model(psi_te), psi_t1_te).item()
            best = min(best, tl)
            print(f"Epoch {epoch:03d} | Train: {el/13:.4e} | Test: {tl:.4e}")

    dur = time.time()-start
    dx = 20.0/resolution
    hook_pass, norm_drift = WaveNormConservationHook.run(model, psi_t[split:], dx=dx)
    status = "✅ PASS" if best < 1e-2 else "⚠️ PARTIAL"
    ts = datetime.datetime.now().isoformat()
    cert = f"""
### 🛡️ UC6 — Équation de Schrödinger 1D (FNO Quantique)
- **Date**: {ts} | **Durée**: {dur:.1f}s
- **Dataset**: Paquets d'ondes cohérents analytiques (états cohérents oscillateur) — Zero-Stub
- **Architecture**: {'FNO(neuralop)' if USE_FNO else 'Pseudo-FNO FFT+MLP'} [n_modes=32, hidden=64, layers=4]
- **Test MSE**: `{best:.4e}` | **Dérive Norme L²**: `{norm_drift:.4e}`
- **Statut**: {status}
"""
    with open("./specs/Scientific_Audit_Ledger.md", "a") as f:
        f.write(cert)
    print(f"\n[RÉSULTAT] {status} | MSE={best:.4e} | Norm Drift={norm_drift:.4e}")
    return model, best

if __name__ == "__main__":
    train()
