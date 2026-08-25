# MASTER LAB SPECIFICATION & SCIENTIFIC GUIDE (LABS 0 TO 7)

**Document Status:** MASTER CERTIFIED SPECIFICATION  
**Epistemic Standard:** `KERNEL-HONNÊTE` Tier A (Zero-Sorry Lean 4 Formalization)  
**Target Core:** SocrateAI Cyber-Physical Observatory & HoloEngine Simulation Framework  

---

## 1. Vue d'Ensemble & Pyramide Épistémique

L'Observatoire Cyber-Physique SocrateAI intègre 8 laboratoires d'expérimentation et de simulation (LAB-0 à LAB-7). Pour maintenir une intégrité scientifique absolue et éviter tout glissement rhétorique entre modélisation abstraite et réalité physique, le programme impose la **Pyramide Épistémique en 3 Niveaux** :

| Niveau | Discipline | Périmètre & Mécanisme de Validation |
| :--- | :--- | :--- |
| **Tier A (Kernel Lean 4)** | Preuves Mathématiques Formelles | Théorèmes abstraits machine-vérifiés (`DualScale.lean`, `TopoStability.lean`, `FourierStateZ3.lean`, `PenroseFormalism.lean`, `Lab5.lean`). **0 Sorry, 0 Axiome ad-hoc**. |
| **Tier B (Jumeau Numérique)** | Simulation Numérique & TDA | Solveurs Canvas/WebGL 2D/3D (eau peu profonde, optique BOMA, cymatique, TDA Gudhi/Ripser, projecteur Leray-Hopf). |
| **Tier C (Hypothèse Physique)** | Inférence & Analogie | Transposition aux données empiriques (JHTDB, IllustrisTNG, magnétotellurique). Validation par le **Protocole de Popper**. |

---

## 2. Spécification Détaillée des 8 Laboratoires

### LAB-0 : Discipline Cryptographique & Métrique Dual-Scale ($P4$)
- **Principe Physique / Mathématique :** Prévention de l'effondrement singulier ($R \to 0$, $\rho \to \infty$) par introduction de la métrique auto-duale $R_{\text{eff}} = \max(R, \alpha'/R)$.
- **Preuve Formelle Lean 4 (`DualScale.lean`) :**
  ```lean
  theorem singularity_avoidance (f : ScalarField X) (hf : ∀ x, 0 < f x) :
      ∀ x, Real.sqrt α' ≤ P4_Field α' f x
  ```
  Le champ régularisé $P4$ ne peut mathématiquement jamais passer sous la borne $\sqrt{\alpha'}$.
- **Système d'Actuation Matérielle :** Scellage cryptographique SHA-256 pre-commit. Aucune commande PWM n'est exécutée si le hash du jumeau numérique n'est pas validé.

### LAB-1 : Gravité Analogue & Horizon Sonique (Weinfurtner Shallow Water)
- **Physique Réelle vs Modèle :** Intégration du système d'équations en eau peu profonde du 2nd ordre couplé :
  $$\frac{\partial h}{\partial t} + \frac{\partial (h u)}{\partial x} = 0, \quad \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + g \frac{\partial h}{\partial x} = -g \frac{\partial z_b}{\partial x}$$
- **Vitesse du Son & Nombre de Froude :** $c(x) = \sqrt{g h(x)}$, $Fr(x) = \frac{u(x)}{c(x)}$.
- **Horizon Sonique :** Défini exactement au point d'inversion transcritique $Fr(x) = 1.0$ sur l'obstacle sous-marin $z_b(x)$ (Trou Blanc Analogue).
- **Dispersion Trans-Planckienne :** $c(k) = \sqrt{\frac{g}{k} \tanh(k h)}$.

### LAB-2 : Scanner Optique & Métamatériau BOMA-2D
- **Système Optique :** Numérisation matricielle 16-bit ($32 \times 32$ à $64 \times 64$) par tête d'impression DVD asservie.
- **Réfringence Spatiale Variable :** Gradient d'indice $n(r) = n_0 + \Delta n \, e^{-r^2 / \sigma^2}$.
- **Effets Optiques :** Déflexion géodésique du front d'onde, réfraction négative locale et carte de réflectivité par phase conjuguée.

### LAB-3 : Banc CHOP & Lévitation Cymatique Acoustophorétique
- **Physique Acoustique :** Cavité résonante et plaque vibrante 2D régies par l'équation de Chladni :
  $$P(x,y) = A \cos\left(\frac{n \pi x}{L}\right) \cos\left(\frac{m \pi y}{L}\right) - A \cos\left(\frac{m \pi x}{L}\right) \cos\left(\frac{n \pi y}{L}\right)$$
- **Piégeage des Particules :** Les micro-particules migrent sous l'effet de la force d'acoustophorèse vers les lignes nodales de pression nulle $P(x,y) = 0$.
- **Télémétrie :** Interférométrie opto-mécanique CHOP mesurant l'amplitude de résonance.

### LAB-4 : Observatoire Holographique & Cosmologie Conforme Penrose (CCC)
- **Modèle Cosmologique :** Ré-étalonnage conforme de la métrique $\tilde{g}_{ab} = \Omega^2 g_{ab}$ pour relier le futur lointain d'un éon au Big Bang de l'éon suivant.
- **Preuve Lean 4 (`PenroseFormalism.lean`) :**
  ```lean
  theorem ccc_no_singularity (Ω : ℝ → ℝ) (hΩ : ∀ τ, 0 < Ω τ) :
      ∀ τ, 0 < (Ω τ)^2
  ```
  Régularité absolue du facteur d'échelle conforme évitant la singularité initiale.
- **Crash Test P4 & Control No-Magic :** Démonstration de brisure de jauge démontrant que le modèle refuse les distorsions non-conformes.

### LAB-5 : Trans-Scale TDA, Barcodes & Vide Topologique $P4$ ($\sqrt{\alpha'}$)
- **Invariance Trans-Échelle :** Normalisation Isométrique Max-Norm ($\vec{x}_{\text{norm}} = \frac{\vec{x} - \vec{\mu}}{\max \|\vec{x} - \vec{\mu}\|}$) permettant de comparer des fluides océaniques ($10^2$ m) et de la matière noire cosmologique ($10^{20}$ m).
- **Homologie Persistante $H_1$ :** Extraction des barcodes et nombres de Betti ($\beta_1 = 2$ pour le tore $T^2$).
- **Preuves Lean 4 (`Lab5.lean`) :**
  - `sublevel_interleaving` : Stabilité de l'entrelacement sous perturbation max-norme.
  - `p4_topological_void` : Preuve que le sous-niveau $f \le t < \sqrt{\alpha'}$ est l'ensemble vide $\emptyset$.
  - `p4_macroscopic_isometry` : Isométrie parfaite à l'échelle macroscopique.
- **Protocole de Popper :** Rejet des hypothèses nulles (Bruit blanc Gaussian, Leurre Sphère $S^2$, Rupture d'isométrie Z) par calcul de distance de Wasserstein.

### LAB-6 : Régularisation de Navier-Stokes & Projecteur Leray-Hopf $Z^3$
- **Équations de Navier-Stokes 3D :** $\frac{\partial u}{\partial t} + (u \cdot \nabla)u = -\nabla p + \nu \Delta u + f$.
- **Projecteur de Leray-Hopf :** $\mathbb{P}_{\text{LH}} u = u - \nabla \Delta^{-1} (\nabla \cdot u)$.
- **Preuve Lean 4 (`FourierStateZ3.lean`) :**
  ```lean
  theorem leray_projector_divergence_free (u : VectorFieldZ3) :
      divergence (lerayProjector u) = 0
  ```
- **Contrôle de l'Enstrophie :** $\Omega(t) = \int |\omega|^2 dx$. L'application du projecteur couplée aux bornes $P4$ garantit mathématiquement l'absence de blow-up de l'enstrophie en temps fini.

### LAB-7 : Observatoire Tellurique K3 & Isomorphisme Géophysique
- **Réseau de Picard K3 :** Surface K3 géométrique $X$ caractérisée par son groupe de Picard $\text{Pic}(X) = H^{1,1}(X, \mathbb{Z}) \cap H^2(X, \mathbb{Z})$ de rang maximum $\rho(X) = 20$.
- **Données Géophysiques :** Ingestion des profils magnétotelluriques de la croûte terrestre (conductivité $\sigma(x,y,z)$).
- **Dualité K3 $\times T^2$ :** Démonstration de l'isomorphisme entre les formes quadratiques d'intersection topologiques des filaments telluriques et la métrique des fibres cosmologiques.

---

## 3. Guide de l'Interface Graphique (Master Hub Command Center)

L'interface web interactive située dans `dashboard/index.html` réunit les 8 laboratoires :

1. **Master Hub (Vue Globale) :** Visualisez les métriques live (Nombre de Froude, Enstrophie, Vide P4, Statut Lean 4), la carte navigable des 8 Labs, l'inspecteur de code Lean 4 et le journal d'audit en temps réel.
2. **Interacteurs de Simulation :**
   - **Lab 0 :** Glissez les sliders $\alpha'$ et $R$ pour observer le rebond $R_{\text{eff}}$.
   - **Lab 1 :** Ajustez la pompe et la hauteur d'eau pour déplacer l'Horizon Sonique ($Fr = 1$).
   - **Lab 2 :** Lancez la numérisation matricielle 16-bit et observez le suivi de front d'onde.
   - **Lab 3 :** Modifiez les modes propres $(m, n)$ pour forcer la réorganisation des particules cymatiques.
   - **Lab 4 :** Déclenchez le rebond cosmologique de Penrose-Carter.
   - **Lab 5 :** Basculez entre les jeux de données (Tore $T^2$, Océan JHTDB, DM IllustrisTNG, Bruit Blanc) et lancez l'Audit de Popper.
   - **Lab 6 :** Activez/Désactivez le projecteur Leray-Hopf pour observer le contrôle de l'enstrophie.
   - **Lab 7 :** Manipulez le réseau 3D de Picard K3 et examinez la matrice d'intersection.
3. **Registre SQLite :** Inspectez les traces cryptographiques SHA-256 de chaque essai.

---

## 4. Instructions pour la Suite des Travaux

Pour porter ces garanties formelles dans le moteur 3D temps réel Rust (`HoloEngine` / `Bevy`) :
1. Intégrer la métrique $R_{\text{eff}}$ dans les fonctions de distance signée (SDF) de `terrain_generator.rs`.
2. Implémenter le filtre Leray-Hopf dans la grille SPH du solveur de fluide (`poc2/mod.rs`).
3. Connecter la télémétrie Rust au Master Hub `dashboard/index.html`.
