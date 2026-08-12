# TNN Univers Model : Catalogue des 10 Cas d'Usages Physiques (Use Cases)

Ce document compile 10 cas d'usages scientifiques couvrant l'ensemble du domaine physique de l'Univers (Mécanique classique, relativité, physique quantique, électromagnétisme, mécanique des fluides, thermodynamique et cosmologie).

---

## 1. Oscillateur Harmonique Couplé (Mécanique Classique)
- **Système** : 2 masses reliées par un ressort.
- **Formulation** : $\mathcal{H}(q, p) = \frac{p_1^2}{2m} + \frac{p_2^2}{2m} + \frac{1}{2}k||q_1 - q_2||^2$.
- **Fichier** : `scripts/train_usecase_spring.py`

## 2. Problème Chaotique des 3 Corps Gravitationnels (Astrophysique)
- **Système** : 3 masses sous attraction gravitationnelle $1/r$.
- **Formulation** : $\mathcal{H}(q, p) = \sum \frac{p_i^2}{2m_i} - \sum_{i<j} \frac{G m_i m_j}{||q_i - q_j||}$.
- **Fichier** : `scripts/train_usecase_3body_gravitation.py`

## 3. Mouvement de Lorentz dans un Champ Électromagnétique (Électromagnétisme)
- **Système** : Particule chargée $q$ dans un champ magnétique $\vec{B}$ et électrique $\vec{E}$.
- **Formulation** : $\vec{F} = q (\vec{E} + \vec{v} \times \vec{B})$.
- **Fichier** : `scripts/train_usecase_lorentz.py`

## 4. Pendule Double Chaotique (Dynamique Non-Linéaire)
- **Système** : 2 tiges rigides sous l'effet de la gravité.
- **Formulation** : Hamiltonien à 2 angles $(\theta_1, \theta_2, p_1, p_2)$ fortement couplé et chaotique.
- **Fichier** : `scripts/train_usecase_double_pendulum.py`

## 5. Gaz Parfait & Distribution de Maxwell-Boltzmann (Thermodynamique Statistique)
- **Système** : Ensemble de $N$ particules d'un gaz en collisions élastiques dans une boîte.
- **Formulation** : Conservation de l'énergie cinétique totale $\sum \frac{1}{2}m v_i^2$ et émergence de la température $T$.
- **Fichier** : `scripts/train_usecase_gas_kinetics.py`

## 6. Équation de Schrödinger Temporelle 1D (Physique Quantique)
- **Système** : Fonction d'onde $\psi(x, t)$ dans un puits de potentiel.
- **Formulation** : $i \hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x)\psi$.
- **Fichier** : `scripts/train_usecase_schrodinger.py`

## 7. Équation de Burgers Visqueuse 1D (Mécanique des Fluides / Chocs)
- **Système** : Champ de vitesse fluide soumis à l'advection et à la viscosité $\nu$.
- **Formulation** : $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$.
- **Fichier** : `scripts/train_usecase_burgers.py`

## 8. Oscillateur Harmonique Relativiste (Relativité Restreinte)
- **Système** : Particule relativiste oscillante à des vitesses proches de la lumière ($c$).
- **Formulation** : $\mathcal{H}(q, p) = \sqrt{p^2 c^2 + m^2 c^4} + \frac{1}{2} k q^2$.
- **Fichier** : `scripts/train_usecase_relativistic.py`

## 9. Équation des Ondes 1D de D'Alembert (Physique Ondulatoire / Électrodynamique)
- **Système** : Propagation d'une onde scalaire dans un milieu continu.
- **Formulation** : $\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$.
- **Fichier** : `scripts/train_usecase_wave_equation.py`

## 10. Expansion Cosmique N-Corps (Cosmologie FLRW)
- **Système** : Distribution de matière subissant l'expansion métrique $a(t)$ de l'Univers.
- **Formulation** : Coordonnées comobiles $x = r/a(t)$ sous l'équation de Friedmann.
- **Fichier** : `scripts/train_usecase_cosmo_flrw.py`
