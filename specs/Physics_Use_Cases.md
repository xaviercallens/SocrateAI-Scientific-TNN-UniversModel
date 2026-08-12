# TNN Univers Model : Cas d'Usages Physiques (Use Cases)

Ce document compile les expérimentations "Use Cases" simples et complexes permettant de valider l'apprentissage des lois fondamentales par le **TNN (Thermodynamic, Topological, Tensor Neural Network)**. 

---

## Cas d'Usage 1 : L'Oscillateur Harmonique Couplé (Système Masse-Ressort à 2 Corps)

### 1. La Physique du Système
Le système étudié est composé de deux particules de masse $m_1 = m_2 = 1$ reliées par un ressort de raideur $k=1$ dans un espace 2D. 

L'état du système est décrit par :
*   Les positions $q_1, q_2 \in \mathbb{R}^2$
*   Les moments $p_1, p_2 \in \mathbb{R}^2$

**Le Hamiltonien (Énergie Totale)** $\mathcal{H}$ est la somme de l'énergie cinétique ($E_k$) et potentielle ($E_p$) :
$$ \mathcal{H}(q, p) = \frac{||p_1||^2}{2m_1} + \frac{||p_2||^2}{2m_2} + \frac{1}{2} k ||q_1 - q_2||^2 $$

---

## Cas d'Usage 2 : Le Problème Chaotique des 3 Corps Gravitationnels (3-Body Problem)

### 1. La Physique du Système
Le problème des 3 corps sous l'effet de la gravité newtonienne est notoirement **chaotique et non intégrable analytiquement**. Une infime perturbation de l'état initial modifie drastiquement les trajectoires futures.

Pour $N=3$ corps de masses $m_i$ aux positions $q_i \in \mathbb{R}^3$ et moments $p_i \in \mathbb{R}^3$ :
$$ \mathcal{H}(q, p) = \sum_{i=1}^3 \frac{||p_i||^2}{2 m_i} - \sum_{1 \le i < j \le 3} \frac{G \, m_i m_j}{||q_i - q_j||} $$

### 2. Le Défi pour les IA Classiques vs TNN
*   **IA Classique (Transformers / Feed-Forward / LSTMs)** : Prédire $(q_{t+1}, p_{t+1})$ par régression directe entraîne une dissipation numérique d'énergie. Les planètes s'effondrent rapidement vers le centre de masse ou sont éjectées artificiellement de leur orbite.
*   **TNN Thermo-Topologique (EGNN + HNN + RK4)** : 
    1.  **EGNN** : Apprend le potentiel gravitationnel $V(q)$ de manière strictement $E(3)$-équivariante (invariante par rotation et translation dans l'espace 3D).
    2.  **HNN** : Dérive les forces exactes via $\dot{p}_i = -\nabla_{q_i} \mathcal{H}$.
    3.  **Intégrateur Symplectique RK4** : Avance l'état temporel tout en préservant le volume dans l'espace des phases (Symplecticités).

### 3. Métriques de Validation
1.  **Stabilité Orbitale à Long Terme** : Zéro effondrement orbital.
2.  **Conservation de l'Énergie** : $\Delta \mathcal{H} \approx 0$ sur 1000 pas de temps.
3.  **Comparaison Baseline** : Comparaison directe contre un réseau dense standard (MLP) sur la même trajectoire.
