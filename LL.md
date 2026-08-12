# Lessons Learned (LL) - TNN Univers Model

Ce document est un registre chronologique des "Lessons Learned" (leçons apprises) lors de la construction du TNN (Thermodynamic, Topological, Tensor Neural Network). Conformément à notre skill `tnn-univers-model`, nous y documentons nos découvertes physiques, architecturales, et la gestion du "Virtual Heat" (overhead computationnel).

---

## Étape 1 : Le MVP Topologique (EGNN et Invariance)

### 1. Ingestion Poly-Algébrique (QM9)
*   **Constat** : Les formats de données classiques (tenseurs plats, images) ne conviennent pas à la physique. Nous avons dû créer le `TNNTopoFormatter` pour isoler les coordonnées purement géométriques ($X \in \mathbb{R}^3$) des caractéristiques intrinsèques (numéro atomique $Z$).
*   **Leçon** : Pour le "Poly-Algebraic Calculus", la matrice d'adjacence (`edge_index`) agit comme le connecteur $N$-aire $\Xi^{\langle N \rangle}$. En rendant le graphe fully-connected, nous ne biaisons pas le réseau avec des liaisons chimiques humaines (simples/doubles), mais nous laissons l'EGNN découvrir les potentiels d'interaction globaux.

### 2. Le EquivarianceHook et la Philosophie "Zero-Sorry"
*   **Constat** : Un réseau classique (MLP) sur des coordonnées $(x, y, z)$ échoue instantanément si on tourne la molécule. Il doit être ré-entraîné (data augmentation). L'EGNN ne requiert aucune augmentation.
*   **Leçon** : Le `EquivarianceHook` prouve qu'un réseau peut "calculer" la physique intrinsèquement. En appliquant une rotation $R \in O(3)$ aléatoire, la prédiction d'énergie ne dévie pas (erreur $\approx 10^{-6}$). C'est ce qu'on appelle la vérification "Zero-Sorry" : l'invariant physique est garanti par l'architecture, pas par l'approximation statistique.

### 3. Overhead Computationnel (Virtual Heat)
*   **Leçon** : Le pooling global (`global_mean_pool`) qui modélise la Poly-Contraction (fusionner $N$ atomes en $1$ scalaire macroscopique) est rapide en PyTorch sur GPU, mais la construction du graphe fully-connected (taille $N \times N$) crée un goulot d'étranglement polynomial. Ce sera notre premier point de friction (Virtual Heat) lors du passage à des systèmes astronomiques (Phase 1 : N-Corps massif). Nous devrons probablement passer à des rayons de coupure locaux pour les grandes échelles.

---

## Étape 2 : Le Moteur Temporel (HNN et Symplecticité)

### 1. Remplacement de l'Architecture Feed-Forward par un Solveur SDE
*   **Constat** : Prédire l'état $t+1$ directement à partir de $t$ via un réseau dense (ex: ResNet) provoque une dissipation d'énergie (le système explose ou s'arrête).
*   **Leçon** : En encapsulant le `Topo-Encoder` à l'intérieur du `HNN` (Hamiltonian Neural Network), le réseau n'apprend plus la dynamique, il apprend **l'énergie totale (le Hamiltonien $\mathcal{H}$)**. Les dérivées temporelles ($\dot{q}, \dot{p}$) sont ensuite calculées mathématiquement par `torch.autograd` (équations de Hamilton). C'est le passage d'une prédiction probabiliste à un calcul physique exact.

### 2. Le SymplecticConservationHook
*   **Constat** : Nous devons nous assurer que le modèle ne viole pas la première loi de la thermodynamique lors des transitions d'états latents.
*   **Leçon** : Le Hook prouve qu'en intégrant $\Delta t$ avec les vecteurs produits par le HNN, le $\Delta E$ est proche de 0 (aux erreurs d'intégration d'Euler près). Si le réseau modifie le système, l'énergie reste bloquée sur la même variété symplectique. 

### 3. Le Goulot de l'Autograd (Virtual Heat - Phase 2)
*   **Leçon** : Le calcul de `torch.autograd.grad` à travers un graphe complet (EGNN) est exponentiellement plus lourd qu'un *forward pass* classique. Nous dépensons beaucoup de "Virtual Heat" (temps CPU/GPU) pour forcer le silicium de von Neumann à respecter la différentiation symplectique, justifiant encore l'utilisation à terme d'architectures matérielles natives (Acoustic-Fluidic vHPU).

---

## Étape 3 : Le Pilier Tensoriel (Champs Continus et Modulus)

### 1. L'Espace des Fréquences via le Fourier Neural Operator (FNO)
*   **Constat** : L'EGNN (N-Corps) s'effondre face à l'infinité des points d'un système fluide (Navier-Stokes). Les GNN ne peuvent pas modéliser l'espace continu.
*   **Leçon** : L'utilisation du `FNO` déplace l'apprentissage dans le domaine de Fourier. Le réseau apprend un opérateur intégral global indépendant de la résolution de la grille (mesh-free). Cela permet au TNN de traiter des grilles 16x16 ou 1024x1024 avec les *mêmes* poids, une nécessité absolue pour le Modèle d'Univers multi-échelle.

### 2. Les Limites de l'Opérateur Pur et le "Physics-Informed" (PINO)
*   **Constat** : Contrairement au HNN qui calcule mathématiquement la physique stricte via l'Autograd, le FNO brut (Vanilla FNO) reste un modèle purement data-driven. Notre `MassConservationHook` démontre qu'à l'initialisation, le champ de vitesse généré viole massivement l'incompressibilité ($\nabla \cdot \vec{v} \neq 0$).
*   **Leçon** : Le FNO doit obligatoirement être transformé en un Physics-Informed Neural Operator (PINO) en utilisant le framework NVIDIA Modulus. Nous devons explicitement définir la pénalité thermodynamique (la conservation de la masse et la dissipation) dans une équation de perte résiduelle pour "forcer" le FNO à obéir aux PDEs de l'Univers. C'est l'essence du futur **Energy Critic**.

---

## Étape 4 : L'Architecture V-JEPA et l'Energy Critic

### 1. De la Coordonnée Physique à l'Espace Latent Canonique
*   **Constat** : Prédire les coordonnées exactes N-corps $(q_{t+1}, p_{t+1})$ dans un fluide ou une galaxie est impossible à cause du chaos déterministe. 
*   **Leçon** : Le V-JEPA résout ce problème en prédisant l'évolution du système **dans l'espace latent**. Le Topo-Encoder compresse la physique (Poly-Contraction) vers un espace canonique réduit $(Q, P)$. C'est dans cet espace abstrait (le *Ruliad*) que le modèle intègre le temps et la thermodynamique de manière hyper-rapide.

### 2. Le Rôle Fondamental de l'Energy Critic
*   **Constat** : Le V-JEPA original de Meta (pour la vidéo ou le texte) utilise la MSE (L2) pure sur les vecteurs latents. Cela provoque souvent un "Dimensional Collapse" ou produit des dynamiques non physiques.
*   **Leçon** : Nous avons remplacé l'erreur L2 par l'**Energy Critic**. Ce dernier force le Prédicteur latent à respecter le principe de conservation d'énergie ($\mathcal{H}(\hat{z}_{t+1}) = \mathcal{H}(z_t)$). Ainsi, l'encodeur ne peut pas tricher en réduisant la variance des vecteurs à zéro ; il est forcé d'apprendre les vraies symétries sous-jacentes du monde physique. Le `RulialInversionHook` confirme la stabilité du système et l'absence d'effondrement dimensionnel.
