# Rapport Détaillé : Bases de Données pour le TNN (Univers Model)

Pour entraîner un **TNN (Thermodynamic, Topological, Tensor Neural Network)** afin qu'il apprenne la dynamique intrinsèque de la nature, les datasets textuels ou d'images 2D classiques sont inutiles. L'architecture nécessite des données purement physiques : des coordonnées spatiales, des tenseurs de vitesse, des énergies potentielles et des structures topologiques.

Voici les meilleures bases de données réelles, libres d'accès et validées scientifiquement pour entraîner les différentes phases de votre "Univers Model".

---

## 1. L'Échelle Macroscopique : Astrophysique et Mécanique Céleste
*Objectif du TNN (Phase 1) : Apprendre la gravité, la conservation du moment cinétique et la dynamique à N-corps via l'EGNN et les réseaux Hamiltoniens.*

*   **IllustrisTNG (The Next Generation)**
    *   **Description** : Les simulations cosmologiques hydrodynamiques les plus avancées au monde, modélisant la formation des galaxies, la matière noire, les champs magnétiques et les trous noirs.
    *   **Format de donnée** : Tenseurs 3D d'hydrodynamique, catalogues de halos et de sous-halos (parfait pour le FNO et Modulus).
    *   **Lien** : [tng-project.org](https://www.tng-project.org/data/)

*   **ESA Gaia Archive (Data Release 3)**
    *   **Description** : Astrométrie de haute précision (positions, distances et mouvements propres) pour près de 2 milliards d'étoiles dans la Voie lactée.
    *   **Format de donnée** : Coordonnées 6D (position + vélocité), idéal pour entraîner le "Topo-Encoder" (e3nn) à découvrir les lois de Kepler et la dynamique galactique.
    *   **Lien** : [gea.esac.esa.int](https://gea.esac.esa.int/archive/)

*   **NASA Exoplanet Archive**
    *   **Description** : Paramètres orbitaux et masses des systèmes exoplanétaires confirmés.
    *   **Lien** : [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu/)

---

## 2. L'Échelle Mésoscopique : Dynamique des Fluides et Systèmes Continus
*Objectif du TNN (Phase 2) : Apprendre les équations de Navier-Stokes, l'incompressibilité, la turbulence et la dissipation via les Fourier Neural Operators (FNO) et l'Energy Critic.*

*   **Johns Hopkins Turbulence Database (JHTDB)**
    *   **Description** : Base de données massive (pétaoctets) contenant les historiques spatio-temporels complets de simulations de turbulence par résolution directe de Navier-Stokes (DNS). C'est le Graal pour comprendre les cascades d'énergie.
    *   **Format de donnée** : Champs de vecteurs de vitesse (3D) et champs scalaires de pression (idéal pour tester la bascule de dimension dans le vHPU).
    *   **Lien** : [turbulence.pha.jhu.edu](http://turbulence.pha.jhu.edu/)

*   **ERA5 (ECMWF - Copernicus)**
    *   **Description** : Réanalyse climatique globale depuis 1940. ERA5 fournit des données horaires sur de nombreux paramètres atmosphériques, terrestres et océaniques.
    *   **Format de donnée** : Grilles spatio-temporelles mondiales de vent, température et pression (le terrain de jeu parfait pour des modèles comme Pangu-Weather ou GraphCast, mais via TNN).
    *   **Lien** : [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)

---

## 3. L'Échelle Microscopique : Physique Quantique et Chimie Topologique
*Objectif du TNN (Phase 3) : Apprendre les interactions fortes/faibles, l'électromagnétisme, la conformation spatiale des atomes via `e3nn`.*

*   **QM9 (Quantum Machines 9)**
    *   **Description** : Propriétés géométriques, énergétiques, électroniques et thermodynamiques de 134 000 petites molécules organiques.
    *   **Format de donnée** : Graphes moléculaires (nœuds = atomes, arêtes = liaisons), avec des cibles telles que l'énergie interne ou l'écart HOMO-LUMO. C'est le dataset de référence pour tester un EGNN.
    *   **Lien** : [quantum-machine.org/datasets](http://quantum-machine.org/datasets/)

*   **MD17 (Molecular Dynamics)**
    *   **Description** : Trajectoires de dynamique moléculaire ab-initio pour diverses molécules organiques.
    *   **Format de donnée** : Séries temporelles de coordonnées 3D atomiques avec les forces et les énergies associées (parfait pour le Lagrangian Neural Network pour l'apprentissage des champs de force).
    *   **Lien** : [sgdml.org](http://www.sgdml.org/)

*   **The Materials Project**
    *   **Description** : Propriétés calculées (band gap, élasticité, piézoélectricité) pour des dizaines de milliers de matériaux cristallins.
    *   **Format de donnée** : Lattices 3D périodiques.
    *   **Lien** : [materialsproject.org](https://materialsproject.org/)

---

## 4. L'Échelle Biologique : Génétique, Protéines et Régulation
*Objectif du TNN (Frontier) : Traiter la biologie sous l'angle topologique, de la forme des protéines à la structure de régulation génétique, en appliquant le "Poly-Algebraic Calculus".*

*   **Protein Data Bank (PDB) & AlphaFold Database**
    *   **Description** : Les archives complètes des structures 3D expérimentales (PDB) et prédites par l'IA (AlphaFold) des protéines.
    *   **Format de donnée** : Nuages de points 3D (coordonnées des acides aminés). La conformation 3D est un problème de minimisation de l'énergie libre, ce qui correspond exactement au cœur du TNN.
    *   **Lien** : [rcsb.org](https://www.rcsb.org/) / [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk/)

*   **Human Cell Atlas (HCA) / Données scRNA-seq**
    *   **Description** : Transcriptomique spatiale et séquençage d'ARN en cellule unique.
    *   **Format de donnée** : Matrices creuses de haute dimension (expression génique). Permet de tester les hyper-graphes topologiques ($\Xi^{\langle N \rangle}$) pour modéliser les relations de co-régulation multi-génique (comme suggéré dans votre roadmap oncologique).
    *   **Lien** : [humancellatlas.org](https://www.humancellatlas.org/)

---
## Conclusion pour l'Ingestion vHPU
Pour le tout premier test de la pipeline complète (du Modulus Data Pipe jusqu'au vHPU en passant par le Topo-Encoder `e3nn` et le Predictor `HNN`), le point de départ idéal est **QM9** (pour valider l'invariance $SE(3)$ au niveau des particules) ou le **JHTDB** (pour valider le cascade d'énergie et la bascule T-Dual via Tensor Operators).
