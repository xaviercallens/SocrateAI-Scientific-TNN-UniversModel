import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

# =============================================================================
# TNN Univers Model : Ingestion QM9 pour le Topo-Encoder
# =============================================================================
# Ce script télécharge et formate la base de données moléculaire QM9.
# Il est conçu pour formater les données spécifiquement pour des réseaux 
# invariants/équivariants (comme EGNN ou e3nn).
# =============================================================================

class TNNTopoFormatter(BaseTransform):
    """
    Un Transformateur PyTorch Geometric pour formater QM9.
    Il prépare les tenseurs nécessaires au Pilier Topologique :
    - Coordonnées 3D (pos)
    - Charges atomiques (z)
    - Énergie interne U0 (notre cible thermodynamique)
    """
    def forward(self, data):
        # 1. Extraction de la structure spatiale (L'Espace)
        # Coordonnées 3D des atomes de la molécule
        pos = data.pos  # Shape: [num_atoms, 3]
        
        # 2. Caractéristiques des noeuds (Atomes)
        # On utilise le numéro atomique (Z) comme feature principale
        atomic_numbers = data.z.unsqueeze(1).float() # Shape: [num_atoms, 1]
        
        # 3. La Cible Thermodynamique (L'Énergie)
        # Dans QM9, l'énergie interne U0 à 0 Kelvin est l'index 7 de data.y
        # On la convertit d'Electron-Volts (eV) à un format tensoriel standard
        u0_energy = data.y[:, 7].unsqueeze(1) # Shape: [1, 1]
        
        # Construction du graphe (Full-connected pour l'EGNN)
        # L'EGNN de Satorras connecte chaque atome à tous les autres
        num_nodes = pos.size(0)
        row = torch.arange(num_nodes, dtype=torch.long)
        col = torch.arange(num_nodes, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        
        # On supprime les auto-boucles (self-loops)
        mask = row != col
        edge_index = edge_index[:, mask]

        # On retourne un objet Data formaté pour notre MVP TNN
        return Data(
            x=atomic_numbers,      # Features : charges
            pos=pos,               # Géométrie : coords 3D
            edge_index=edge_index, # Topologie : Graphe
            energy_target=u0_energy# Cible : U0
        )

def get_qm9_dataloader(root_dir='./data/QM9', batch_size=32, split='train'):
    """
    Télécharge, formate et retourne un DataLoader PyTorch pour QM9.
    """
    print(f"[*] Initialisation de la base de données QM9 dans {root_dir}...")
    
    # Le dataset QM9 natif de PyTorch Geometric gère le téléchargement auto.
    dataset = QM9(root=root_dir, transform=TNNTopoFormatter())
    
    # Shuffle et séparation Train/Test
    # (Par convention pour QM9 : 100k train, 18k validation, 13k test)
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    
    if split == 'train':
        dataset = dataset[:100000]
    elif split == 'val':
        dataset = dataset[100000:118000]
    elif split == 'test':
        dataset = dataset[118000:]
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    print(f"[+] Dataset '{split}' chargé. Nombre de molécules : {len(dataset)}")
    return loader

if __name__ == "__main__":
    import torch_geometric # Assurez-vous d'avoir installé pip install torch_geometric
    
    os.makedirs("./data", exist_ok=True)
    
    # Test d'ingestion
    try:
        train_loader = get_qm9_dataloader(split='train')
        
        # Inspecter le premier batch
        for batch in train_loader:
            print("\n--- TNN Topo-Batch Inspection ---")
            print(f"Batch de graphes (nb molécules = {batch.num_graphs})")
            print(f"Matrice des Noeuds (Features Atomes) : {batch.x.shape}")
            print(f"Matrice de Géométrie (Coords 3D)   : {batch.pos.shape}")
            print(f"Matrice Topologique (Edge Index)   : {batch.edge_index.shape}")
            print(f"Vecteur Énergie Cible (U0)         : {batch.energy_target.shape}")
            break
            
        print("\n[SUCCESS] Ingestion réussie. Prêt à être connecté à l'EGNN.")
        
    except ImportError:
        print("[ERROR] La bibliothèque 'torch_geometric' n'est pas installée.")
        print("Exécutez : pip install torch-geometric")
