📚 Boîte à Outils pour l'Implémentation du TNN (Univers Model)
Ce document liste les articles scientifiques (ArXiv) et les dépôts GitHub open-source les plus pertinents pour construire les trois piliers du TNN : Thermodynamique (Énergie), Topologique (Géométrie), et Tensoriel (Continu), ainsi que l'architecture World Model.
1. Pilier Thermodynamique & Énergie (Le Moteur)
Pour apprendre à votre réseau à conserver l'énergie et à respecter le principe de moindre action, vous devez vous baser sur les "Réseaux Hamiltoniens" et "Lagrangiens".
📄 Articles Fondateurs
"Hamiltonian Neural Networks" (Greydanus et al., 2019) - ArXiv: 1906.01563
Pourquoi le lire : C'est LE papier qui a prouvé qu'un réseau de neurones pouvait apprendre exactement les lois de conservation de l'énergie à partir de données brutes.
"Lagrangian Neural Networks" (Miles Cranmer et al., 2020) - ArXiv: 2003.04630
Pourquoi le lire : Plus général que les HNN, il permet d'apprendre la physique sans avoir besoin de coordonnées canoniques (idéal pour des systèmes complexes).
💻 Repositories GitHub
greydanus/hamiltonian-nn
Ce que vous y trouverez : Le code PyTorch original du papier HNN. Parfait pour extraire la brique de calcul de la dérivée énergétique (comme dans le script précédent).
MilesCranmer/lagrangian_nns
Ce que vous y trouverez : L'implémentation PyTorch des réseaux Lagrangiens. Miles Cranmer est une référence dans l'IA appliquée à l'astrophysique.
2. Pilier Topologique (L'Espace et les Symétries)
Pour manipuler des atomes, des molécules ou des systèmes solaires dans un espace 3D de manière native (sans que l'IA ne soit perturbée par la rotation ou la translation de l'univers).
📄 Articles Fondateurs
"E(n) Equivariant Graph Neural Networks" (Satorras et al., 2021) - ArXiv: 2102.09844
Pourquoi le lire : C'est la base de l'architecture EGNN. Indispensable pour la phase "Quantique/Moléculaire" de votre roadmap.
"Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds" (Thomas et al., 2018) - ArXiv: 1802.08219
Pourquoi le lire : C'est le fondement de l'utilisation des tenseurs purs pour respecter la géométrie 3D de la physique.
💻 Repositories GitHub
vgsatorras/egnn
Ce que vous y trouverez : Le code officiel pour modéliser des interactions de N-corps. À intégrer directement comme votre "Topo-Encoder" pour les données discrètes (particules).
e3nn/e3nn (Euclidean neural networks)
Ce que vous y trouverez : C'est LA librairie PyTorch de référence (soutenue par le LBNL et MIT) pour créer des réseaux de neurones qui respectent nativement la géométrie de l'espace 3D. C'est une brique incontournable pour votre TNN.
3. Pilier Tensoriel & Continu (Les Fluides et Champs)
Pour simuler la dynamique des fluides, le climat, et tout ce qui est modélisé par des équations aux dérivées partielles (EDP).
📄 Articles Fondateurs
"Fourier Neural Operator for Parametric Partial Differential Equations" (Li et al., 2020) - ArXiv: 2010.08895
Pourquoi le lire : Caltech a révolutionné l'IA scientifique avec ce papier en créant les opérateurs neuronaux qui apprennent dans l'espace de Fourier (indépendant de la résolution).
💻 Repositories GitHub
neuraloperator/neuraloperator
Ce que vous y trouverez : La bibliothèque PyTorch de Caltech. C'est le meilleur point de départ pour remplacer les réseaux denses par des opérateurs tensoriels dans votre Bloc Prédicteur.
NVIDIA/modulus
Ce que vous y trouverez : Le framework open-source massif de NVIDIA dédié au Physics-ML. Modulus contient des PINNs, des FNOs, et des PINO (Physics-Informed Neural Operators) hyper optimisés pour GPU. Vous devriez probablement baser toute votre infrastructure de bas niveau sur Modulus.
4. L'Inspiration "World Model" (L'Architecture Globale)
Pour structurer l'apprentissage via l'Énergie et la prédiction d'états latents, comme recommandé par Yann LeCun.
📄 Articles Fondateurs
"A Path Towards Autonomous Machine Intelligence" (Yann LeCun, 2022)
Lien : Papier de position disponible via Meta AI (OpenReview).
Pourquoi le lire : Le manifeste fondateur de l'architecture JEPA (Joint Embedding Predictive Architecture). C'est votre bible pour designer la boucle d'apprentissage (Predictor + Critic) sans utiliser de texte/LLM.
"V-JEPA: Video Joint Embedding Predictive Architecture" (Bardes et al., Meta AI, 2024)
💻 Repositories GitHub
facebookresearch/jepa et facebookresearch/v-jepa
Ce que vous y trouverez : L'implémentation officielle de Meta. Bien que ce soit pour la vidéo et l'image, vous pouvez cloner le design de la "Loss" (l'Energy Critic) et remplacer leurs encodeurs d'images par vos encodeurs physiques (e3nn ou FNO).
🛠️ Plan d'Action pour l'Intégration (Hackathon interne)
Si vous devez monter un prototype rapidement, voici la "chimère" architecturale idéale :
Installez NVIDIA Modulus : Utilisez-le comme framework de base pour la gestion des données physiques et l'entraînement GPU.
Codez le "Topo-Encoder" avec e3nn : Transformez vos données brutes (ex: positions du système solaire) en vecteurs invariants.
Codez le "Predictor" avec lagrangian_nns : Utilisez le réseau de Miles Cranmer pour calculer l'état latent futur.
Enveloppez le tout dans l'architecture V-JEPA (Meta) : Pour l'entraînement auto-supervisé, demandez au réseau de prédire la dynamique future en cachant (masking) certaines parties du système physique.
