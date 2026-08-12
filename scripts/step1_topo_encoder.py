import sys
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# Ajout du chemin pour importer l'EGNN cloné
sys.path.append('./reference_repos/egnn')
from models.egnn_clean.egnn_clean import EGNN

class TNNTopoEncoder(nn.Module):
    """
    Étape 1 du TNN Univers Model : Le MVP Topologique
    Utilise un EGNN pour apprendre les caractéristiques invariantes d'un système
    physique (N-corps ou molécules).
    """
    def __init__(self, in_node_nf=1, hidden_nf=64, out_node_nf=64, n_layers=4):
        super().__init__()
        # Le cœur Poly-Algébrique (qui manipule les variables Xi<N>)
        self.egnn = EGNN(in_node_nf=in_node_nf, hidden_nf=hidden_nf, out_node_nf=out_node_nf, 
                         in_edge_nf=0, n_layers=n_layers)
        
        # La tête de prédiction thermodynamique
        self.energy_mlp = nn.Sequential(
            nn.Linear(out_node_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
    def forward(self, h, x, edges, batch_index):
        # 1. Mise à jour équivariante des caractéristiques et des coordonnées
        h_out, x_out = self.egnn(h, x, edges, edge_attr=None)
        
        # 2. Agrégation Poly-Contraction : on rassemble les nœuds pour former la macro-molécule
        h_mol = global_mean_pool(h_out, batch_index)
        
        # 3. Prédiction de la valeur scalaire (ex: U0)
        energy = self.energy_mlp(h_mol)
        return energy, x_out

# =============================================================================
# HOOKS TNN
# =============================================================================
def get_random_rotation_matrix():
    """Génère une matrice de rotation 3D aléatoire (O(3))."""
    q, r = torch.linalg.qr(torch.randn(3, 3))
    return q * torch.sign(torch.diag(r))

def EquivarianceHook(model, batch):
    """
    Hook TNN : Zéro-Sorry Verification pour la Topologie.
    Applique une rotation aléatoire à l'univers et vérifie que la prédiction
    d'énergie (scalaire) reste strictement invariante.
    """
    print("\n[HOOK] Lancement du EquivarianceHook (Vérification SE(3))...")
    model.eval()
    with torch.no_grad():
        # Prédiction sur l'univers de base
        energy_base, _ = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        
        # Rotation de l'univers
        R = get_random_rotation_matrix()
        pos_rotated = torch.matmul(batch.pos, R.T)
        
        # Prédiction sur l'univers tourné
        energy_rotated, _ = model(batch.x, pos_rotated, batch.edge_index, batch.batch)
        
        # Calcul de la violation
        max_diff = torch.abs(energy_base - energy_rotated).max().item()
        
        if max_diff < 1e-4:
            print(f"[HOOK] ✅ PASS : Le système est topologiquement invariant (Violation = {max_diff:.3e})")
            return True
        else:
            print(f"[HOOK] ❌ FAIL : Violation d'invariance géométrique (Violation = {max_diff:.3e})")
            return False

if __name__ == "__main__":
    from ingest_qm9 import get_qm9_dataloader
    
    loader = get_qm9_dataloader(split='train', batch_size=32)
    batch = next(iter(loader))
    
    print("\n--- Initialisation du Topo-Encoder ---")
    model = TNNTopoEncoder()
    
    # Exécution du Hook pour valider le modèle avant tout entraînement
    EquivarianceHook(model, batch)
