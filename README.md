# TNN Univers Model (Thermodynamic, Topological, Tensor Neural Network)

**Univers Model** est une architecture d'Intelligence Artificielle de nouvelle génération (Physics-ML). Contrairement aux Large Language Models (LLMs) entraînés sur du texte, l'Univers Model est un **V-JEPA (Joint-Embedding Predictive Architecture)** entraîné exclusivement sur les lois physiques de l'univers.

## 🏗 L'Architecture Chimère : Les 3 Piliers

L'Univers Model repose sur une conception tri-partite (Poly-Algebraic Calculus) qui remplace les encodeurs standards (ResNets, Transformers) :

1. **Pilier Topologique (L'Espace)** :
   - **Rôle** : Modélise la géométrie stricte et les invariants spatiaux sans data-augmentation.
   - **Technologie** : $E(n)$-Equivariant Graph Neural Networks (EGNN).
   - **Fichier** : `scripts/step1_topo_encoder.py`
   - **Validation** : `EquivarianceHook` (Vérifie la conservation sous symétrie $SE(3)$).

2. **Pilier Thermodynamique (Le Temps)** :
   - **Rôle** : Calcule les champs de vecteurs temporels et intègre la dynamique N-Corps de façon symplectique.
   - **Technologie** : Hamiltonian Neural Networks (HNN).
   - **Fichier** : `scripts/step2_thermo_predictor.py`
   - **Validation** : `SymplecticConservationHook` (Vérifie la stricte conservation de l'énergie, $\Delta E = 0$).

3. **Pilier Tensoriel (Le Continu)** :
   - **Rôle** : Modélise l'infinité des champs continus (Navier-Stokes, turbulence) sans dépendre de la résolution de la grille.
   - **Technologie** : Fourier Neural Operators (FNO) / Modulus.
   - **Fichier** : `scripts/step3_tensor_fluid.py`
   - **Validation** : `MassConservationHook` (Vérifie l'incompressibilité du fluide, $\nabla \cdot \vec{v} = 0$).

## 🚀 Installation & Exécution

Assurez-vous d'avoir installé les dépendances via `pip` ou dans un environnement virtuel :
```bash
pip install torch torch-geometric imageio tensorly tensorly-torch
```

Testez l'intégrité "Zero-Sorry" de chaque pilier :
```bash
# Tester l'invariance SE(3) sur des molécules QM9
python scripts/step1_topo_encoder.py

# Tester la conservation de l'énergie sur un système N-Corps
python scripts/step2_thermo_predictor.py

# Tester la modélisation de fluide et mesurer la divergence
python scripts/step3_tensor_fluid.py
```

## 📖 Documentation

*   Consultez le fichier `LL.md` (Lessons Learned) pour le journal d'implémentation et de la théorie (Virtual Heat, Zero-Sorry Verification).
*   Consultez le dossier `specs/` pour la roadmap, les manifestes mathématiques et la bibliographie.
