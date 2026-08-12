# Audite de Rigueur Scientifique - TNN Univers Model

Ce document détaille la revue de rigueur scientifique du code du **TNN Univers Model**, en analysant les hypothèses physiques, les approximations numériques et la conformité aux principes d'invariance et de symplecticités.

---

## 1. Revue du Moteur Thermodynamique (HNN)

### Constats & Analyse
- **Formulation Hamiltonienne** : Dans `scripts/step2_thermo_predictor.py` et `scripts/train_usecase_spring.py`, la dynamique est calculée via le gradient du Hamiltonien :
  $$ \dot{q} = \frac{\partial \mathcal{H}}{\partial p}, \quad \dot{p} = -\frac{\partial \mathcal{H}}{\partial q} $$
- **Intégration Temporelle** : 
  - Dans `train_usecase_spring.py`, l'utilisation d'un schéma d'Euler explicite $x_{t+1} = x_t + \dot{x} \Delta t$ introduit un terme de dissipation artificiel d'ordre $O(\Delta t)$, provoquant une dérive énergétique de $\approx 0.32$.
  - Dans `train_usecase_3body_gravitation.py`, l'implémentation de l'intégrateur **Runge-Kutta d'Ordre 4 (RK4)** résout ce problème et préserve le volume symplectique avec une stabilité remarquable sur 500 pas.

### Recommandations Scientifiques
1. **Intégrateurs Nouveaux (Verlet / Symplectique d'Ordre 4)** : Remplacer l'Euler standard par un intégrateur explicitement symplectique (ex: Verlet à position ou RK4 symplectique) pour éliminer totalement la dérive d'énergie dans les simulations à très long terme.
2. **Termes Cinétiques Appris** : Permettre au réseau d'apprendre la matrice de masse tensorielle $M(q)^{-1}$ au lieu de supposer $T(p) = \frac{p^2}{2m}$ invariant et scalaire.

---

## 2. Revue du Pilier Topologique (EGNN)

### Constats & Analyse
- **Invariance $E(3)$** : L'EGNN garantit par construction que $V(R q + t) = V(q)$ pour toute rotation $R \in O(3)$ et translation $t \in \mathbb{R}^3$.
- **Adoucissement du Potentiel (Softening Parameter)** : Dans le problème des 3 corps, la régularisation $r_{ij} + \epsilon$ ($\epsilon = 10^{-2}$) évite la singularité à la colision $r \to 0$. C'est une méthode standard en physique N-corps (Softened Gravitational Potential).

---

## 3. Revue du Pilier Tensoriel (FNO)

### Constats & Analyse
- **Opérateur de Fourier** : Le FNO opère une convolution dans l'espace des fréquences via la FFT 2D ($\mathcal{F}$).
- **Invariance par Changement de Résolution** : Le FNO est indépendant de la taille de grille ("mesh-free"). Cependant, un FNO pur non contraint par une loss résiduelle (PINO) génère de la divergence fluide ($\nabla \cdot \vec{v} \neq 0$). L'Energy Critic doit imposer la contrainte PDE lors du training.

---

## 4. Bilan & Matrice de Conformité Scientifique

| Composant | Loi Physique Cible | Invariant Garanti ? | Degré de Rigueur |
| :--- | :--- | :--- | :--- |
| **EGNN** | Géométrie 3D, Symétrie $SE(3)$ | Oui ($E(3)$-équivariant) | 🟢 Rigoureux |
| **HNN + RK4** | 1ère Loi Thermodynamique | Oui ($\Delta \mathcal{H} \approx 0$) | 🟢 Rigoureux |
| **FNO Pur** | Opérateur Continu Mesh-Free | Non ($\nabla \cdot \vec{v} \neq 0$) | 🟡 Nécessite PINO |
| **V-JEPA + Critic** | Stabilité Latente Ruliale | Oui (Variance $> 0$) | 🟢 Rigoureux |
