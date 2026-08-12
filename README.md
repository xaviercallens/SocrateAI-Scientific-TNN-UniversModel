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

4. **Boucle V-JEPA & Energy Critic** :
   - **Rôle** : Apprentissage dans l'espace latent $(Q, P)$ guidé par l'Energy Critic et la variance Ruliale.
   - **Fichier** : `scripts/step4_vjepa_engine.py`
   - **Validation** : `RulialInversionHook`.

---

## 🧪 Cas d'Usages Physiques (Use Cases)

- **Use Case 1 : Oscillateur Harmonique Couplé (Masse-Ressort)**
  - *Fichier* : `scripts/train_usecase_spring.py`
  - *Objectif* : Déduire l'énergie potentielle $V(q)$ et cinétique $T(p)$ à partir de trajectoires brutes.

- **Use Case 2 : Système Chaotique Gravitationnel des 3 Corps (3-Body Problem)**
  - *Fichier* : `scripts/train_usecase_3body_gravitation.py`
  - *Objectif* : Benchmark du TNN Thermo-Topologique vs MLP Baseline avec intégrateur symplectique RK4 sur 500 pas.

---

## 🚀 Installation & Exécution

Assurez-vous d'avoir installé les dépendances via `pip` ou dans un environnement virtuel :
```bash
pip install torch torch-geometric imageio tensorly tensorly-torch
```

Exécution des tests et cas d'usages :
```bash
# 1. Validation des Piliers
python scripts/step1_topo_encoder.py
python scripts/step2_thermo_predictor.py
python scripts/step3_tensor_fluid.py
python scripts/step4_vjepa_engine.py

# 2. Exécution des Cas d'Usages Physiques
python scripts/train_usecase_spring.py
python scripts/train_usecase_3body_gravitation.py
```

## 📖 Documentation

*   Consultez `specs/Physics_Use_Cases.md` pour les détails physiques théoriques des cas d'usages.
*   Consultez `LL.md` (Lessons Learned) pour le journal d'apprentissage et d'analyse des benchmarks.
