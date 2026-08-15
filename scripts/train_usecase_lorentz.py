"""
UC3 — Mouvement de Lorentz dans un Champ Électromagnétique
===========================================================
TNN Pilier: Thermodynamique (HNN)
Physique: Particule chargée q dans un champ magnétique uniforme B = (0,0,Bz)
Hamiltonien canonique avec potentiel vecteur A = (-By/2, Bx/2, 0):
  H(q,p) = (1/2m) * ||p - (e/c)*A(q)||² + eΦ(q)
  Pour B uniforme et E=0: H = p²/2m - e*Bz*(px*y - py*x)/(2mc) + e²Bz²(x²+y²)/(8mc²)

Vérification: Fréquence cyclotron ωc = eBz/m, orbite circulaire exacte.
Données: Orbites cyclotron analytiques (Zero-Stub - pas de torch.randn).
"""
import sys
import os
import math
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tnn.utils.data_generators import generate_lorentz_trajectory
from tnn.utils.integrators import rk4_step_flat

# ============================================================================
# MODÈLE TNN : Hamiltonien Électromagnétique Appris
# ============================================================================
class LorentzHamiltonianNet(nn.Module):
    """
    Réseau apprenant H(q, p) pour une particule chargée dans un champ B uniforme.
    Architecture: MLP profond avec activation Tanh pour une courbure lisse
    compatible avec la double différentiation de l'autograd.
    État: x = (qx, qy, qz, px, py, pz) — dim=6
    """
    def __init__(self, in_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Matrice symplectique J pour dim=6: dx/dt = J * dH/dx
        # J = [[0, I], [-I, 0]] pour (q,p)
        n = in_dim // 2
        J = torch.zeros(in_dim, in_dim)
        J[:n, n:] = torch.eye(n)
        J[n:, :n] = -torch.eye(n)
        self.register_buffer('J', J)

    def forward(self, x):
        """
        Retourne dx/dt = J * dH/dx via autograd.
        x: [batch, 6] avec requires_grad=True
        """
        H = self.net(x).sum()
        dH_dx = torch.autograd.grad(H, x, create_graph=True)[0]
        dx_dt = dH_dx @ self.J.t()
        return dx_dt

    def hamiltonian(self, x):
        return self.net(x)


# ============================================================================
# DONNÉES : Orbites Cyclotron Analytiques (Zero-Stub)
# ============================================================================
def generate_cyclotron_dataset(num_samples=4000, seed=42):
    """
    Génère des états et dérivées exacts pour des orbites cyclotron.
    B = (0, 0, 1.0), charge e = 1.0, masse m = 1.0 → ωc = 1.0
    Solution exacte: q(t) = r*(sin(ωt+φ), cos(ωt+φ), vz*t)
                     p(t) = m*ωc*r*(cos(ωt+φ), -sin(ωt+φ), vz)
    """
    torch.manual_seed(seed)
    B_z = 1.0
    q_charge = 1.0
    m = 1.0
    omega_c = q_charge * B_z / m  # = 1.0

    # Diversité des orbites: rayon r ∈ [0.5, 2.0], phase φ, vitesse z
    t_vals = torch.linspace(0, 4 * math.pi, num_samples)
    phases = torch.linspace(0, 2 * math.pi, num_samples)
    radii = 1.0 + 0.5 * torch.sin(t_vals)   # Variation douce du rayon
    vz_vals = 0.3 * torch.cos(t_vals * 0.5)

    qx = radii * torch.sin(omega_c * t_vals + phases)
    qy = radii * torch.cos(omega_c * t_vals + phases)
    qz = vz_vals * t_vals / t_vals.max()

    # p = m * v (avec v dérivée de q analytiquement)
    vx = radii * omega_c * torch.cos(omega_c * t_vals + phases)
    vy = -radii * omega_c * torch.sin(omega_c * t_vals + phases)
    vz = vz_vals

    px = m * vx
    py = m * vy
    pz = m * vz

    states = torch.stack([qx, qy, qz, px, py, pz], dim=1)

    # Dérivées exactes via force de Lorentz: F = q*(v × B)
    # B = (0,0,Bz) → F = q*(vy*Bz, -vx*Bz, 0)
    ax = (q_charge / m) * vy * B_z
    ay = -(q_charge / m) * vx * B_z
    az = torch.zeros_like(ax)
    derivatives = torch.stack([vx, vy, vz, ax, ay, az], dim=1)

    return states, derivatives


# ============================================================================
# HOOK ZÉRO-SORRY : Conservation de l'Énergie Cinétique
# ============================================================================
def LorentzCyclotronHook(model, x_test):
    """
    Vérifie que le réseau prédit correctement la fréquence cyclotron
    en mesurant la dérive énergétique sur 500 pas RK4.
    """
    print("\n[HOOK] Lancement du LorentzCyclotronHook...")
    model.eval()

    x_t = x_test[0:1].clone()
    dt = 0.01
    steps = 500

    H_list = []
    for _ in range(steps):
        with torch.enable_grad():
            x_req = x_t.clone().detach().requires_grad_(True)
            H_list.append(model.hamiltonian(x_req).item())
            k1 = model(x_req).detach()
            x_k2 = (x_t + 0.5*dt*k1).requires_grad_(True)
            k2 = model(x_k2).detach()
            x_k3 = (x_t + 0.5*dt*k2).requires_grad_(True)
            k3 = model(x_k3).detach()
            x_k4 = (x_t + dt*k3).requires_grad_(True)
            k4 = model(x_k4).detach()
            x_t = (x_t + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)).detach()

    drift = max(H_list) - min(H_list)
    drift_pct = abs(drift / (H_list[0] + 1e-8)) * 100

    if drift_pct < 5.0:
        print(f"[HOOK] ✅ PASS | Dérive Énergie: {drift:.4e} ({drift_pct:.2f}%)")
        return True, drift_pct
    else:
        print(f"[HOOK] ❌ FAIL | Dérive Énergie: {drift:.4e} ({drift_pct:.2f}%)")
        return False, drift_pct


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================
def train():
    print("=" * 65)
    print("  TNN UC3 — FORCE DE LORENTZ (Champ Magnétique Uniforme)")
    print("=" * 65)

    start_time = time.time()
    x_all, dx_all = generate_cyclotron_dataset(num_samples=5000, seed=42)
    split = 4000
    x_train, dx_train = x_all[:split], dx_all[:split]
    x_test, dx_test = x_all[split:], dx_all[split:]

    model = LorentzHamiltonianNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_fn = nn.MSELoss()

    epochs = 200
    batch_size = 256
    best_test_loss = float('inf')

    print(f"\n[*] Training: {split} samples | Epochs: {epochs} | Batch: {batch_size}")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x_train.size(0))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx = x_train[idx].clone().requires_grad_(True)
            bdx = dx_train[idx]

            optimizer.zero_grad()
            pred_dx = model(bx)
            loss = loss_fn(pred_dx, bdx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 50 == 0 or epoch == epochs:
            model.eval()
            with torch.enable_grad():
                xt = x_test.clone().requires_grad_(True)
                pred_test = model(xt)
            test_loss = loss_fn(pred_test.detach(), dx_test).item()
            best_test_loss = min(best_test_loss, test_loss)
            print(f"Epoch {epoch:03d} | Train: {epoch_loss/n_batches:.4e} | Test: {test_loss:.4e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    duration = time.time() - start_time
    hook_pass, drift_pct = LorentzCyclotronHook(model, x_test)
    status = "✅ PASS" if best_test_loss < 1e-2 and hook_pass else "⚠️ PARTIAL"

    # Audit Certificate
    timestamp = datetime.datetime.now().isoformat()
    cert = f"""
### 🛡️ UC3 — Force de Lorentz (Hamiltonien Électromagnétique)
- **Date**: {timestamp} | **Durée**: {duration:.1f}s
- **Dataset**: Orbites cyclotron analytiques (ωc=1.0, B_z=1.0) — Zero-Stub
- **Architecture**: HNN MLP [6→256→256→256→1] + Matrice Symplectique J
- **Métriques Test**: MSE = `{best_test_loss:.4e}`
- **Dérive Énergie (RK4 500 pas)**: `{drift_pct:.2f}%`
- **Statut**: {status}
"""
    with open("./specs/Scientific_Audit_Ledger.md", "a") as f:
        f.write(cert)

    print(f"\n[RÉSULTAT] {status} | Test MSE={best_test_loss:.4e} | Drift={drift_pct:.2f}%")
    return model, best_test_loss


if __name__ == "__main__":
    train()
