import sys
import torch
import torch.nn as nn

# Ajout au chemin pour importer le FNO
sys.path.insert(0, './reference_repos/neuraloperator')
from neuralop.models.fno import FNO

class TNNTensorEncoder(nn.Module):
    """
    Étape 3 : Pilier Tensoriel (Continue/Fluides).
    Utilise le Fourier Neural Operator (FNO) pour prédire un champ vectoriel (ex: Navier-Stokes).
    Ici on modélise l'espace continu via des tenseurs en domaine de Fourier.
    """
    def __init__(self, resolution=(16, 16)):
        super().__init__()
        # In_channels = 1 (Vorticité ou champ scalaire initial)
        # Out_channels = 2 (Champ de vitesse vectoriel (u, v) pour 2D)
        # On utilise une factorisation très légère (rank=0.1) typique du TFNO pour gagner en vitesse
        self.fno = FNO(
            n_modes=(8, 8), 
            in_channels=1, 
            out_channels=2, 
            hidden_channels=32, 
            n_layers=4,
            projection_channel_ratio=1,
            lifting_channel_ratio=1
        )
        
    def forward(self, x):
        # x : [batch, in_channels, x_res, y_res]
        return self.fno(x)

# =============================================================================
# HOOKS TNN
# =============================================================================
def MassConservationHook(model, fluid_state):
    """
    Hook TNN : Vérification de la loi de conservation de masse (incompressibilité).
    Pour un fluide incompressible (Navier-Stokes), la divergence du champ de vitesse doit être nulle.
    ∇·V = (du/dx) + (dv/dy) = 0
    """
    print("\n[HOOK] Lancement du MassConservationHook (Test d'Incompressibilité)...")
    model.eval()
    
    with torch.no_grad():
        # Vitesse prédite : [batch, 2, x_res, y_res]
        velocity_field = model(fluid_state)
        
        u = velocity_field[:, 0, :, :]
        v = velocity_field[:, 1, :, :]
        
        # Calcul de la divergence par différences finies (espacement dx=1, dy=1 simplifié)
        du_dx = u[:, 1:, :] - u[:, :-1, :]
        dv_dy = v[:, :, 1:] - v[:, :, :-1]
        
        # Pour superposer les grilles, on rogne le dernier pixel
        du_dx = du_dx[:, :, :-1]
        dv_dy = dv_dy[:, :-1, :]
        
        divergence = du_dx + dv_dy
        
        # Calcule l'erreur moyenne de divergence
        mean_div = torch.abs(divergence).mean().item()
        
        print(f"       -> Divergence moyenne : {mean_div:.4e}")
        
        # Dans un FNO brut, la physique n'est pas forcée. 
        # C'est ici que Modulus (PINO) interviendra en ajoutant la divergence à la loss (Energy Critic).
        if mean_div < 1e-2:
            print("[HOOK] ✅ PASS : Le fluide est globalement incompressible (divergence faible).")
            return True
        else:
            print("[HOOK] ⚠️ WARNING : Haute divergence (Non-Incompressible) !")
            print("       Un réseau FNO pur est 'Physics-Agnostic'. Il doit être couplé au Physics-Informed")
            print("       Energy Critic (PINO/Modulus) pour minimiser explicitement cette divergence ∇·V=0.")
            return False

if __name__ == "__main__":
    import math
    print("--- Initialisation du Tensor Encoder (Pilier Continu) ---")
    print("    [Zero-Stub] Utilisation d'un champ de vorticité spectral déterministe")
    model = TNNTensorEncoder()
    
    # Zero-Stub Policy: Deterministic spectral vorticity field (NOT torch.randn)
    # Physically-motivated: superposition of Fourier modes representing
    # an incompressible 2D Taylor-Green vortex
    x = torch.linspace(0, 2 * math.pi, 16)
    y = torch.linspace(0, 2 * math.pi, 16)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    fluid_state = (torch.sin(2 * grid_x) * torch.cos(2 * grid_y) +
                   0.3 * torch.sin(4 * grid_x)).unsqueeze(0).unsqueeze(0)
    
    MassConservationHook(model, fluid_state)
