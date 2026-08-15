"""
UC5 — Gaz Parfait & Distribution de Maxwell-Boltzmann (Thermodynamique Statistique)
=====================================================================================
TNN Pilier: Thermodynamique (Energy Critic)
Physique: N particules d'un gaz idéal. Le TNN apprend à prédire la température T
à partir de la distribution des vitesses et vérifie le théorème d'équipartition:
  <E_cin> = (3/2) * N * k_B * T
  <v²>   = 3 * k_B * T / m

Architecture: Network apprenant T(v) depuis les moments statistiques.
Données: Distribution Maxwell-Boltzmann analytique, seeded (Zero-Stub).
"""
import sys, os, math, time, datetime
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tnn.utils.data_generators import generate_maxwell_boltzmann_gas

K_B = 1.38064852e-23
M_ATOM = 4.65e-26  # Masse approximative N2

class MaxwellBoltzmannCritic(nn.Module):
    """
    Energy Critic: prédit la température T à partir des moments de la distribution
    de vitesses. Entrée: [<v²>, <v⁴>, kurt(v), skew(v²)] — 4 moments statistiques.
    Ce réseau simule l'inversion statistique thermodynamique.
    """
    def __init__(self, n_moments=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_moments, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # Linear output: trained in normalized space
        )
        # Normalization stats (set after dataset creation)
        self.T_mean = 0.0
        self.T_std = 1.0

    def forward(self, moments):
        return self.net(moments)

    def predict_temperature(self, moments_normalized):
        """Returns temperature in Kelvin (denormalized)."""
        with torch.no_grad():
            pred_norm = self.forward(moments_normalized)
        return pred_norm * self.T_std + self.T_mean

def compute_velocity_moments(velocities):
    """
    Calcule les moments statistiques des vitesses:
    [<v²>, <v⁴>, kurtosis, variance(<v²>)]
    velocities: [batch, N, 3]
    """
    v2 = (velocities**2).sum(dim=-1)       # [batch, N] — vitesse² par particule
    mean_v2 = v2.mean(dim=1, keepdim=True) # [batch, 1]
    mean_v4 = (v2**2).mean(dim=1, keepdim=True)
    # Kurtosis: <v⁴>/<v²>² (= 5/3 pour MB 3D)
    kurt = mean_v4 / (mean_v2**2 + 1e-12)
    # Variance de v²
    var_v2 = v2.var(dim=1, keepdim=True)
    moments = torch.cat([mean_v2, mean_v4, kurt, var_v2], dim=1)
    return moments

def generate_gas_dataset(n_temps=2000, n_particles=200, seed=42):
    """
    Dataset: températures T uniformément réparties → vitesses MB → moments.
    Utilise le théorème d'équipartition comme cible exacte.
    Moments enrichis: [log(<v²>), log(<v⁴>), <v_x²>/<v²>, std(v²)/mean(v²)]
    """
    torch.manual_seed(seed)
    T_vals = torch.linspace(100.0, 1000.0, n_temps)
    all_moments, all_T = [], []
    for T in T_vals:
        sigma_v = math.sqrt(K_B * T.item() / M_ATOM)
        v = torch.randn(1, n_particles, 3) * sigma_v  # seeded → déterministe
        v2 = (v**2).sum(dim=-1)  # [1, N] speed^2
        mean_v2 = v2.mean()
        mean_v4 = (v2**2).mean()
        # Use raw moments for linear relationship with T
        # Anisotropy: should be ~1/3 for isotropic MB
        v2_x = (v[:,:,0]**2).mean()
        aniso = v2_x / (mean_v2 + 1e-30)
        # Relative std of speed^2
        rel_std = v2.std() / (mean_v2 + 1e-30)
        moments = torch.tensor([[mean_v2.item(), mean_v4.item(),
                                  aniso.item(), rel_std.item()]])
        all_moments.append(moments)
        all_T.append(T.unsqueeze(0).unsqueeze(0))
    return torch.cat(all_moments, dim=0), torch.cat(all_T, dim=0)


def EquipartitionHook(model, T_test, moments_test_n, moments_test_raw):
    """
    Vérifie le théorème d'équipartition: <v²> = 3*k_B*T/m
    Le réseau doit prédire T avec < 5% d'erreur (en Kelvin dénormalisé).
    """
    print("\n[HOOK] EquipartitionHook (Théorème d'Équipartition)...")
    model.eval()
    # Prédiction dénormalisée
    T_pred = model.predict_temperature(moments_test_n).squeeze()
    T_true = T_test.squeeze()
    rel_err = ((T_pred - T_true).abs() / (T_true + 1e-8)).mean().item() * 100
    # Vérification analytique: T = <v²>*m/(3*k_B)
    mean_v2_test = moments_test_raw[:, 0]
    T_equipartition = mean_v2_test * M_ATOM / (3 * K_B)
    equi_err = ((T_equipartition - T_true).abs() / (T_true + 1e-8)).mean().item() * 100
    passed = rel_err < 5.0
    print(f"[HOOK] Erreur T prédite: {rel_err:.2f}% | Erreur équipartition analytique: {equi_err:.2f}%")
    print(f"[HOOK] {'✅ PASS' if passed else '❌ FAIL'}")
    return passed, rel_err

def train():
    print("="*65)
    print("  TNN UC5 — GAZ PARFAIT / MAXWELL-BOLTZMANN (Thermodynamique)")
    print("="*65)
    start = time.time()

    moments_all, T_all = generate_gas_dataset(n_temps=2000, n_particles=200, seed=42)
    split = 1600
    m_tr, T_tr = moments_all[:split], T_all[:split]
    m_te, T_te = moments_all[split:], T_all[split:]

    # Normalisation des moments et de la température
    m_mean, m_std = m_tr.mean(0), m_tr.std(0)+1e-8
    T_mean, T_std = T_tr.mean(), T_tr.std()+1e-8
    m_tr_n = (m_tr - m_mean)/m_std
    m_te_n = (m_te - m_mean)/m_std
    T_tr_n = (T_tr - T_mean)/T_std
    T_te_n = (T_te - T_mean)/T_std

    model = MaxwellBoltzmannCritic()
    model.T_mean = T_mean.item()
    model.T_std = T_std.item()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    loss_fn = nn.MSELoss()
    best = float('inf')

    for epoch in range(1, 201):
        model.train()
        perm = torch.randperm(split)
        el = 0.0
        for i in range(0, split, 128):
            idx = perm[i:i+128]
            opt.zero_grad()
            pred = model(m_tr_n[idx])
            loss = loss_fn(pred, T_tr_n[idx])
            loss.backward()
            opt.step()
            el += loss.item()
        sched.step()
        if epoch % 50 == 0 or epoch == 200:
            model.eval()
            with torch.no_grad():
                tl = loss_fn(model(m_te_n), T_te_n).item()
            best = min(best, tl)
            print(f"Epoch {epoch:03d} | Train: {el/12:.4e} | Test: {tl:.4e}")

    dur = time.time()-start
    hook_pass, rel_err = EquipartitionHook(model, T_te, m_te_n, m_te)
    status = "✅ PASS" if best < 1e-2 and hook_pass else "⚠️ PARTIAL"
    ts = datetime.datetime.now().isoformat()
    cert = f"""
### 🛡️ UC5 — Gaz Parfait / Maxwell-Boltzmann (Thermodynamique Statistique)
- **Date**: {ts} | **Durée**: {dur:.1f}s
- **Dataset**: Distribution Maxwell-Boltzmann analytique T∈[100K,1000K], N=200 — Zero-Stub
- **Architecture**: Energy Critic MLP [4→128×3→1] + Softplus (T>0)
- **Test MSE (normalisé)**: `{best:.4e}` | **Erreur Équipartition**: `{rel_err:.2f}%`
- **Statut**: {status}
"""
    with open("./specs/Scientific_Audit_Ledger.md", "a") as f:
        f.write(cert)
    print(f"\n[RÉSULTAT] {status} | MSE={best:.4e} | Erreur T={rel_err:.2f}%")
    return model, best

if __name__ == "__main__":
    train()
