# INVENTAIRE COMPLET DES TNN ET DATASETS PRÉ-ENTRAÎNÉS (LAB-0 À LAB-7)

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
| **LAB-4** | `tnn_holographic_p4_detector.py` | Holographic Bottleneck TNN ($\chi = 8$) | Loi d'Aire & Rebond P4 ($r_h$) | `tnn_lab4_holographic_p4.pt` |
| **LAB-5** | `tnn_lab5_tda.py` | Persistent Homology PointNet | Entrelacement Max-Norm & Wasserstein | `tnn_lab5_tda.pt` |
| **LAB-6** | `tnn_lab6_navier_stokes.py` | Leray-Hopf Z3 Projection Net | Borne Enstrophie (Divergence Nulle) | `tnn_lab6_navier_stokes.pt` |
| **LAB-7** | `tnn_lab7_k3.py` | Telluric K3 Oracle Autoencoder | Isomorphisme Matrice Intersection | `tnn_lab7_k3.pt` |

---

## 2. Jeux de Données Ouverts & Benchmarks (`./data/open_datasets/`)

1. **`openpiv_vortex_benchmark.json`** : Données de vélocimétrie PIV issues de *Weinfurtner et al. (2011)* pour la cinématique de vortex d'Unruh.
2. **`pinn_hydrodynamic_benchmark.json`** : Profils de Froude et de profondeur de canal issus de *Raissi et al. (2019)* pour la physique informée.
3. **`tda_jhtdb_vortex.json`** : Nuage de points TDA pour l'analyse des sous-niveaux d'isométrie macroscopique.

*Certifié par l'Observatoire SocrateAI Master Hub.*
