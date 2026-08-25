# TNN Univers Model (Thermodynamic, Topological, Tensor Neural Network)

> **Naming caveat (N-1):** "Topological" refers to the EGNN/e3nn equivariant pillar and Lab5 TDA pipeline. Individual benchmarks (e.g., Lab1 ResConv1D) may not use topological structures — see [`NAMING_POLICY.md`](NAMING_POLICY.md) for scope.

**Univers Model** est une architecture d'Intelligence Artificielle de nouvelle génération (Physics-ML). Contrairement aux Large Language Models (LLMs) entraînés sur du texte, l'Univers Model est un **V-JEPA (Joint-Embedding Predictive Architecture)** entraîné exclusivement sur les lois physiques de l'univers.

---

## 🏆 Benchmark Multi-Univers (10/10 Cas d'Usages Validés)

Le TNN a été rigoureusement audité et validé sous 1% d'erreur (`< 1e-2`) à travers **10 cas d'usages scientifiques** couvrant l'ensemble de la physique de l'Univers :

| # | Domaine Physique | Cas d'Usage | Modèle / Opérateur | Statut |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Mécanique Classique** | Mass-Spring 2D | Hamiltonian NN (HNN) | ✅ PASS (`9.50e-5`) |
| 2 | **Astrophysique** | 3-Body Gravitational Problem | EGNN + HNN + RK4 | ✅ PASS (`7.58e-4`) |
| 3 | **Électromagnétisme** | Lorentz Force $\vec{F} = q(\vec{E} + \vec{v} \times \vec{B})$ | Vector Field Predictor | ✅ PASS (`7.99e-5`) |
| 4 | **Dynamique Non-Linéaire** | Pendule Double Chaotique | Non-linear HNN | ✅ PASS (`3.30e-4`) |
| 5 | **Thermodynamique Stat.** | Maxwell-Boltzmann Ideal Gas | Kinetic Energy Critic | ✅ PASS (`3.86e-3`) |
| 6 | **Physique Quantique** | Schrödinger 1D Wavefunction $\psi(x,t)$ | Complex Hamiltonian | ✅ PASS (`2.25e-3`) |
| 7 | **Mécanique des Fluides** | Burgers Visqueux 1D | Fourier Operator (FNO) | ✅ PASS (`3.38e-3`) |
| 8 | **Relativité Restreinte** | Oscillateur Relativiste $E=\sqrt{p^2 c^2 + m^2 c^4}$ | Relativistic Energy | ✅ PASS (`2.63e-4`) |
| 9 | **Électrodynamique** | Équation des Ondes d'Alembert $\nabla^2 u$ | Wave Operator | ✅ PASS (`1.97e-3`) |
| 10 | **Cosmologie** | Expansion FLRW $a(t)$ | Friedmann Equation | ✅ PASS (`1.30e-3`) |

---

## 🏗 L'Architecture Chimère : Les 3 Piliers

1. **Pilier Topologique (L'Espace)** : $E(n)$-Equivariant Graph Neural Networks (EGNN).
2. **Pilier Thermodynamique (Le Temps)** : Hamiltonian Neural Networks (HNN).
3. **Pilier Tensoriel (Le Continu)** : Fourier Neural Operators (FNO) / Modulus.
4. **Boucle V-JEPA & Energy Critic** : Apprentissage dans l'espace latent $(Q, P)$ guidé par l'Energy Critic.

---

## 🚀 Installation & Exécution

```bash
pip install torch torch-geometric imageio tensorly tensorly-torch
```

Lancer la suite de benchmark complète des 10 cas d'usages :
```bash
python scripts/train_usecase_suite_10.py
```

Exécuter les benchmarks individuels :
```bash
python scripts/train_usecase_spring.py
python scripts/train_usecase_3body_gravitation.py
```

---

## 📖 Documentation

*   `specs/Scientific_Rigor_Audit.md` : Audite de rigueur scientifique et symplecticités.
*   `specs/Physics_Use_Cases.md` : Formulation théorique des 10 cas d'usages.
*   `LL.md` : Registre des Lessons Learned.
