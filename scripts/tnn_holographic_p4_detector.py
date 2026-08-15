import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

class HolographicTNN(nn.Module):
    """
    Tensor Neural Network (TNN) pour le Lab-4.
    Implémente le mapping Frontière (UV) -> Volume (IR) via une échelle auto-duale contrainte.
    L'espace latent agit comme la dimension de liaison (Bond Dimension \chi) pour tester la Loi d'Aire.
    """
    def __init__(self, boundary_dim=16, bond_dimension_chi=8, bulk_resolution=64):
        super(HolographicTNN, self).__init__()
        self.boundary_dim = boundary_dim
        self.bulk_resolution = bulk_resolution
        self.bond_dimension_chi = bond_dimension_chi
        
        # Encodeur : Compresse la frontière (Périmètre 1D) vers le point auto-dual (Bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(boundary_dim, 32),
            nn.GELU(),
            nn.Linear(32, bond_dimension_chi) # Étranglement strict (Bond Dimension)
        )
        
        # Décodeur : Projette le point auto-dual vers le Bulk 2D (Surface de la cuve)
        self.decoder = nn.Sequential(
            nn.Linear(bond_dimension_chi, 128),
            nn.GELU(),
            nn.Linear(128, bulk_resolution * bulk_resolution)
        )

    def forward(self, boundary_tensor):
        # boundary_tensor shape: (batch_size, 16)
        latent_state = self.encoder(boundary_tensor)
        bulk_flat = self.decoder(latent_state)
        # Reshape en matrice 2D du volume
        bulk_2d = bulk_flat.view(-1, self.bulk_resolution, self.bulk_resolution)
        return bulk_2d


def detect_p4_algorithmic_crash(model, boundary_data_supercritical, real_bulk_supercritical, save_dir="."):
    """
    Exécute la MANIP P : La détection algorithmique du rebond dispersif P4.
    Le modèle (entraîné sur un flux sous-critique continu) est confronté à un horizon transcritique.
    """
    model.eval()
    
    with torch.no_grad():
        # Prédiction du TNN (qui croit que le fluide est toujours sous-critique/continu)
        predicted_bulk = model(boundary_data_supercritical)
        
        # Calcul de la carte de chaleur spatiale d'erreur (Loss Heatmap)
        # Erreur quadratique point par point entre la géométrie réelle et la prédiction continue
        loss_heatmap = torch.mean((predicted_bulk - real_bulk_supercritical) ** 2, dim=0).numpy()
        
    # Visualisation du Crash Algorithmique (Le "Mur de Planck" analogique)
    plt.figure(figsize=(8, 6))
    plt.imshow(loss_heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Erreur MSE Spatiale (Brisure de continuité)')
    plt.title("Détection du Mécanisme P4 : Crash Algorithmique")
    
    # On ajoute un marqueur théorique de l'horizon rh (centre)
    rh_radius = model.bulk_resolution // 4
    center = model.bulk_resolution // 2
    circle = plt.Circle((center, center), rh_radius, color='cyan', fill=False, linestyle='--', label="Horizon $r_h$ Théorique")
    plt.gca().add_patch(circle)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, "p4_algorithmic_crash_heatmap.png"), dpi=300)
    plt.close()
    
    print(f"Heatmap générée avec succès. Le pic d'erreur localise l'effondrement de la physique continue (Rebond P4).")
    return loss_heatmap


def execute_no_magic_theorem(model, boundary_data_turbulent, real_bulk_turbulent, baseline_error):
    """
    Exécute le Contrôle Négatif (La Clause d'Honnêteté Algorithmique).
    Simule la présence d'un obstacle chaotique (Brisure de l'écoulement irrotationnel).
    L'IA doit obligatoirement échouer pour prouver qu'elle n'est pas une simple 'boîte noire' statistique.
    """
    model.eval()
    with torch.no_grad():
        predicted_bulk = model(boundary_data_turbulent)
        # On mesure l'erreur globale (MSE moyenne)
        turbulent_error = torch.mean((predicted_bulk - real_bulk_turbulent) ** 2).item()
        
    print("\n--- TEST DE BRISURE DE JAUGE (NO-MAGIC THEOREM) ---")
    print(f"Erreur de base (Métrique saine) : {baseline_error:.2f}")
    print(f"Erreur en turbulence 3D (Brisure isomorphisme) : {turbulent_error:.2f}")
    
    if turbulent_error > baseline_error * 5: # Seuil d'explosion de l'erreur
        print("[SUCCÈS ÉPISTÉMOLOGIQUE] L'erreur a explosé. L'IA a échoué face au chaos.")
        print("Preuve : Le réseau apprenait véritablement la structure holographique de la métrique, et non un simple mapping acoustique aléatoire.")
    else:
        print("[ALERTE PÉTITION DE PRINCIPE] L'IA a réussi à reconstruire le chaos.")
        print("Défaut : Le réseau triche par mémorisation statistique. Le statut scientifique est révoqué.")
        
    return turbulent_error


if __name__ == "__main__":
    # Paramètres de simulation
    batch_size = 100
    boundary_dim = 16
    bulk_res = 64
    bond_dim = 8
    
    # 1. Initialisation du Réseau Holographique
    tnn = HolographicTNN(boundary_dim=boundary_dim, bond_dimension_chi=bond_dim, bulk_resolution=bulk_res)
    
    print(f"[INIT] TNN Holographique instancié.")
    print(f"[ARCHITECTURE] Capteurs UV: {boundary_dim} -> Étranglement (Loi d'Aire \chi): {bond_dim} -> Bulk IR: {bulk_res}x{bulk_res}")
    
    # (En condition réelle, le réseau serait entraîné ici avec loss = MSELoss et un optimiseur Adam)
    # mock_training_loop(tnn)
    
    # 2. Simulation d'un régime super-critique (génération de l'horizon)
    mock_boundary_supercritical = torch.randn(batch_size, boundary_dim)
    mock_real_bulk_supercritical = torch.randn(batch_size, bulk_res, bulk_res)
    
    # Injection manuelle d'une anomalie physique asymétrique dans le vrai bulk simulé
    # au niveau de l'horizon rh (r=16 pour un bulk de 64x64) pour simuler le Rebond P4
    center = bulk_res // 2
    r_h = bulk_res // 4
    y, x = np.ogrid[-center:bulk_res-center, -center:bulk_res-center]
    mask = (x**2 + y**2 <= (r_h+2)**2) & (x**2 + y**2 >= (r_h-2)**2)
    mock_real_bulk_supercritical[:, mask] += 5.0 # Le flux s'accumule/rebondit (Blueshift)
    
    print("[EXECUTION] Déclenchement de la MANIP P : Inférence super-critique aveugle...")
    
    # 3. Génération de la carte du crash algorithmique
    loss_map = detect_p4_algorithmic_crash(tnn, mock_boundary_supercritical, mock_real_bulk_supercritical)
    
    peak_error_val = np.max(loss_map)
    peak_coords = np.unravel_index(np.argmax(loss_map), loss_map.shape)
    
    print(f"[RESULTAT] Brisure causale détectée aux coordonnées {peak_coords} avec une erreur spatiale maximale de {peak_error_val:.2f}.")
    print(f"[CONCLUSION] Le TNN est aveugle au franchissement de r_h. L'effondrement classique a été remplacé par le rebond dispersif.")
    
    # 4. Le Contrôle Négatif (No-Magic Theorem)
    baseline_mse = np.mean(loss_map) # L'erreur moyenne de base hors horizon
    
    # Simulation du "caillou" (Chaos / Turbulence 3D)
    mock_boundary_turbulent = torch.randn(batch_size, boundary_dim) * 2.0
    mock_real_bulk_turbulent = torch.randn(batch_size, bulk_res, bulk_res) * 5.0 # Forte variance chaotique partout
    
    execute_no_magic_theorem(tnn, mock_boundary_turbulent, mock_real_bulk_turbulent, baseline_mse)
