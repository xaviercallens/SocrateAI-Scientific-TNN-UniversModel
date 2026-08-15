import sys
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# Ajouts au chemin pour importer les modules clonés (priorité à hamiltonian-nn pour éviter la collision sur 'utils.py')
sys.path.insert(0, './reference_repos/egnn')
sys.path.insert(0, './reference_repos/hamiltonian-nn')

from models.egnn_clean.egnn_clean import EGNN
from hnn import HNN

class TopoHamiltonian(nn.Module):
    """
    Réseau Differentiable qui calcule le Hamiltonien H(q, p).
    Il combine l'EGNN (pour l'énergie potentielle topologique)
    et une branche classique (pour l'énergie cinétique).
    """
    def __init__(self, hidden_nf=64):
        super().__init__()
        # Le Topo-Encoder (Gère les positions q)
        # in_node_nf = 1 (masse ou charge)
        self.egnn = EGNN(in_node_nf=1, hidden_nf=hidden_nf, out_node_nf=hidden_nf, in_edge_nf=0)
        self.potential_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
    def forward(self, x_state, edges, batch_index, mass):
        # x_state est de dimension [N, 6] : 3 pour q (position), 3 pour p (moment)
        q, p = torch.split(x_state, 3, dim=1)
        
        # 1. Énergie Potentielle (Topologique via EGNN)
        h, _ = self.egnn(mass, q, edges, edge_attr=None)
        h_mol = global_mean_pool(h, batch_index)
        E_potential = self.potential_mlp(h_mol).sum() # Somme sur le batch
        
        # 2. Énergie Cinétique classique : p^2 / 2m
        # (Dans un vrai TNN, le réseau pourrait l'apprendre, mais ici on l'encode explicitement 
        # pour forcer la physique exacte sur la partie cinétique).
        E_kinetic = torch.sum((p ** 2) / (2 * mass))
        
        # Le Hamiltonien Total H = T + V
        H = E_kinetic + E_potential
        
        # Le HNN attend une sortie de dimension [batch, 2] (Conservative, Solenoidal)
        # H correspond au champ Solenoidal (F2).
        # On duplique la valeur pour matcher la signature requise par le code de Greydanus.
        out = torch.stack([H, H]).unsqueeze(0) # [1, 2]
        return out

class TNNThermodynamicPredictor(nn.Module):
    """
    Étape 2 : Le Moteur Temporel.
    Intègre le Topo-Hamiltonien dans le HNN pour prédire la dynamique temporelle.
    """
    def __init__(self):
        super().__init__()
        self.topo_hamiltonian = TopoHamiltonian()
        
        # Enveloppe HNN : calcule (dq/dt, dp/dt) = (dH/dp, -dH/dq)
        # input_dim = 6 (3 pour q, 3 pour p)
        self.hnn = HNN(input_dim=6, differentiable_model=self._hnn_wrapper, field_type='solenoidal')
        
        self.edges = None
        self.batch_index = None
        self.mass = None
        
    def _hnn_wrapper(self, x_state):
        # Wrapper pour injecter la topologie (edges) dans la signature stricte du HNN
        return self.topo_hamiltonian(x_state, self.edges, self.batch_index, self.mass)
        
    def forward(self, x_state, edges, batch_index, mass):
        self.edges = edges
        self.batch_index = batch_index
        self.mass = mass
        
        # time_derivative renvoie dx/dt = [dq/dt, dp/dt]
        dx_dt = self.hnn.time_derivative(x_state)
        return dx_dt

# =============================================================================
# HOOKS TNN
# =============================================================================
def SymplecticConservationHook(model, q0, p0, edges, batch_index, mass, dt=1e-3):
    """
    Hook TNN : Zéro-Sorry Verification pour la Thermodynamique.
    Intègre le système de t à t+dt et vérifie que l'énergie totale (Hamiltonien)
    est strictement conservée.
    """
    print("\n[HOOK] Lancement du SymplecticConservationHook (Vérification de l'Énergie)...")
    
    # L'état doit requérir le gradient pour l'autograd du HNN
    x0 = torch.cat([q0, p0], dim=1).requires_grad_(True)
    
    # 1. Calcul de l'énergie initiale E_0
    model.edges = edges; model.batch_index = batch_index; model.mass = mass
    H_0 = model.topo_hamiltonian(x0, edges, batch_index, mass)[0, 1].item()
    
    # 2. Prédiction de la dynamique (dx/dt)
    dx_dt = model(x0, edges, batch_index, mass)
    
    # 3. Intégration temporelle simple (Euler) -> t+1
    # Note: En réalité on utiliserait RK4 (Runge-Kutta 4) ou un intégrateur symplectique
    x1 = x0 + dx_dt * dt
    
    # 4. Calcul de l'énergie à t+1
    H_1 = model.topo_hamiltonian(x1, edges, batch_index, mass)[0, 1].item()
    
    delta_E = abs(H_0 - H_1)
    
    if delta_E < 1e-3:
         print(f"[HOOK] ✅ PASS : L'énergie totale est conservée (ΔE = {delta_E:.4e})")
         print(f"       -> Énergie Initiale : {H_0:.4f} | Énergie t+1 : {H_1:.4f}")
         return True
    else:
         print(f"[HOOK] ❌ FAIL : Violation de la conservation d'énergie ! (ΔE = {delta_E:.4e})")
         return False

if __name__ == "__main__":
    # Zero-Stub Policy: Deterministic 2-body spring oscillator initial conditions
    N = 2
    print("--- Initialisation du Thermodynamic Predictor (2-Corps) ---")
    print("    [Zero-Stub] Utilisation de conditions initiales déterministes (ressort)")
    model = TNNThermodynamicPredictor()
    
    # Deterministic initial conditions: two masses on a spring axis
    # Body 1 at (+1, 0, 0), Body 2 at (-1, 0, 0)
    q0 = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    # Momenta: Body 1 moving up, Body 2 moving down (oscillatory)
    p0 = torch.tensor([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    mass = torch.ones(N, 1)  # Masses unitaires
    
    # Graphe fully connected (2-corps)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_index = torch.zeros(N, dtype=torch.long)
    
    SymplecticConservationHook(model, q0, p0, edges, batch_index, mass, dt=1e-4)
