# INVENTAIRE COMPLET DES TNN ET DATASETS PRÉ-ENTRAÎNÉS (LAB-0 À LAB-4)

**Statut :** TNN Spécialisés Initialisés et Poids Sauvegardés  
**Espace de Stockage :** `./models/` & `./data/open_datasets/`

---

## 1. Modèles Tensoriels Spécialisés (TNNs)

| Laboratoire | Fichier du Modèle | Architecture TNN | Invariant Preservé / Loss | Poids Sauvegardés |
| :--- | :--- | :--- | :--- | :---: |
| **LAB-0** | `tnn_lab0_symplectic.py` | Symplectic Hamiltonian Neural Network | Conservation de l'Énergie $(q, p)$ | `tnn_lab0_symplectic.pt` |
| **LAB-1** | `tnn_lab1_fourier.py` | Complex Spectral Optical Correlator | Filtrage Dual Space Fourier | `tnn_lab1_fourier.pt` |
| **LAB-2** | `tnn_lab2_boma2d.py` | Neural PIV / Conv2D Decoder | Reconstruction Champ de Vitesse (u, v) | `tnn_lab2_boma2d.pt` |
| **LAB-3** | `tnn_lab3_hydro_pinn.py` | Hydrodynamic PINN (Shallow Water) | Bernoulli $v \partial_x v + g \partial_x h = 0$ | `tnn_lab3_hydro_pinn.pt` |
| **LAB-4** | `tnn_holographic_p4_detector.py` / `retrain_lab4_holographic_tnn.py` | Holographic Bottleneck TNN ($\chi = 8$) + Pretrained RAFT Optical Flow | Loi d'Aire & Rebond P4 ($r_h$) (MSE: $0.0078$, Ratio Turbulence: $4666\times$) | `tnn_lab4_holographic_p4_retrained.pt` + `raft_small_C_T_V2.pth` |

---

## 2. Jeux de Données Ouverts & Benchmarks (`./data/open_datasets/`)

1. **`openpiv_vortex_benchmark.json`** : Données de vélocimétrie PIV issues de *Weinfurtner et al. (2011)* pour la cinématique de vortex d'Unruh.
2. **`pinn_hydrodynamic_benchmark.json`** : Profils de Froude et de profondeur de canal issus de *Raissi et al. (2019)* pour la physique informée.

*Certifié par l'Observatoire SocrateAI.*
