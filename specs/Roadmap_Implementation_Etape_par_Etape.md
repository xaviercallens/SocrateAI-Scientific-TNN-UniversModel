# Roadmap d'Implémentation Progressive du TNN (Univers Model)

L'objectif de cette roadmap est de construire l'Univers Model étape par étape, en partant de briques open-source existantes (MVP) et en introduisant la complexité (Poly-Algebraic Calculus, vHPU, V-JEPA) de manière itérative.

## Étape 1 : Le MVP Topologique (L'Espace statique)
**Objectif** : Valider que le réseau comprend la géométrie 3D sans tenir compte du temps.
*   **Dataset** : **QM9** (134k molécules organiques). C'est un dataset statique parfait pour valider l'invariance spatiale.
*   **Composant** : Le **Topo-Encoder** (En utilisant l'architecture `EGNN` de Satorras).
*   **Tâche** : Prédire l'énergie interne $U_0$ d'une molécule à partir des coordonnées 3D de ses atomes.
*   **Ce qu'on réutilise** : Le code de `reference_repos/egnn/qm9`.
*   **Hook TNN introduit** : `EquivarianceHook`. Un hook PyTorch appelé à chaque itération qui applique une matrice de rotation aléatoire $R \in SO(3)$ aux coordonnées d'entrée et s'assure que la prédiction de l'énergie (scalaire) reste strictement identique : $f(Rx) = f(x)$.

## Étape 2 : Le Moteur Thermodynamique (Le Temps)
**Objectif** : Introduire la dynamique temporelle et la conservation de l'énergie.
*   **Dataset** : **MD17** (Dynamique moléculaire) ou un simulateur **N-Corps** basique (généré synthétiquement avec les lois de Newton).
*   **Composants** : Topo-Encoder (`EGNN`) couplé au **Thermodynamic Predictor** (`HNN` de Greydanus).
*   **Tâche** : Prédire la position et la vélocité des particules à $t+1$.
*   **Ce qu'on réutilise** : L'autograd du Hamiltonien `reference_repos/hamiltonian-nn/hnn.py` branché sur les embeddings du `EGNN`.
*   **Hook TNN introduit** : `SymplecticConservationHook`. Un hook qui calcule l'énergie totale (Cinétique + Potentielle) à l'instant $t$ et $t+1$ et déclenche une alerte si $\Delta E > \epsilon$ (violation des lois de la thermodynamique).

## Étape 3 : Le Pilier Tensoriel (Les Champs Continus)
**Objectif** : Gérer la dynamique des fluides et les systèmes continus indépendamment de la grille.
*   **Dataset** : **Navier-Stokes / JHTDB** (Données de turbulence 2D/3D via Modulus).
*   **Composants** : Le **Tensor Encoder** (En utilisant le `FNO` de Caltech).
*   **Tâche** : Prédire le champ de vorticité d'un fluide à $t+10$.
*   **Ce qu'on réutilise** : NVIDIA Modulus pour l'ingestion des PDEs et `reference_repos/neuraloperator`.
*   **Hook TNN introduit** : `MassConservationHook`. S'assure que la divergence du champ de vitesse est nulle ($\nabla \cdot \vec{v} = 0$) pour un fluide incompressible.

## Étape 4 : L'Univers Model et la V-JEPA (La Chimère Finale)
**Objectif** : Unifier le modèle sous-jacent avec un apprentissage auto-supervisé.
*   **Dataset** : Multi-modal (Cosmologie + Fluides + Molécules).
*   **Composants** : L'architecture de boucle **JEPA** remplaçant la MSE Loss par l'**Energy Critic**.
*   **Tâche** : Masking temporel et spatial. On donne l'état de l'univers à $t$, on masque une région, et le réseau doit générer la dynamique latente de cette région.
*   **Hook TNN introduit** : `RulialInversionHook`. Le pont final avec le vHPU. Si la densité d'énergie d'un graphe approche la limite $\sqrt{\alpha'}$, ce hook force la bascule topologique T-Dual pour éviter la singularité.
