# Phase 2: RunuX TNN Integration Roadmap (The Neuro-Symbolic Bridge)

## Objectif Principal
Remplacer l'environnement d'exécution PyTorch (instable, gourmand en "Virtual Heat" à cause du GIL Python) par le **RunuX AI Runtime**. Ce passage vers un noyau Rust `no_std` va permettre d'exploiter la sécurité mémoire native, d'atteindre 88% d'occupation sur les Google TPU v5e/v6e (via `tpu_pjrt`), et de lier formellement les axiomes mathématiques du noyau **Lean 4** via une FFI (Foreign Function Interface).

## Étape 2.1 : Mise en place du pont Neuro-Symbolique (Lean 4 ↔ Rust FFI)
*   **Action** : Compiler les théorèmes de `specs/TNN_Invariants.lean` (Conservation Symplectique de l'Énergie) vers des librairies C statiques.
*   **Action** : Développer le module Rust `runux-ai-runtime/crates/ai_runtime/src/lean_ffi.rs` pour lier le noyau d'exécution aux preuves générées.
*   **Livrable** : Le "Zero-Sorry Symplectic Hook" implémenté en Rust natif, rejetant au niveau du CPU/TPU toute prédiction (EGNN/FNO) qui viole la thermodynamique.

## Étape 2.2 : Portage du Pilier Topologique (EGNN) sur Rust/RVV
*   **Action** : Réécrire l'Encodeur Topologique (EGNN) en utilisant le crate `rvv_simd` de RunuX.
*   **Action** : Optimiser le passage de messages (Message Passing) N-corps pour utiliser les instructions vectorielles RISC-V 256/1024-bit.
*   **Livrable** : L'Hémisphère Gauche (Logique/Spatial) de l'architecture *SymBrain v4*, capable de simuler la topologie 3D (MD17/QM9) sans overhead mémoire.

## Étape 2.3 : Tiling Systolique du Pilier Tensoriel (FNO) pour TPU
*   **Action** : Traduire l'opérateur continu FNO (Transformées de Fourier Rapides 2D) en utilisant `stablehlo` et `tpu_pjrt`.
*   **Action** : Forcer le Tiling Systolique à 128x128 (via `mlgo_advisor`) pour maximiser la bande passante HBM (High Bandwidth Memory) sur le Cloud TPU.
*   **Livrable** : L'Hémisphère Droit (Synthétique/Ondulatoire) pour simuler Navier-Stokes et la Cosmologie FLRW avec des performances "bare-metal".

## Étape 2.4 : Implémentation du Routeur PFC (V-JEPA & HNN)
*   **Action** : Implémenter le V-JEPA et le HNN Energy Critic au sein de la boucle principale d'inférence Rust.
*   **Action** : Utiliser le compresseur `turbo_quant` de RunuX pour compresser les grilles EDP de 4096 dimensions vers les 64 dimensions latentes.
*   **Livrable** : Un moteur exécutable (`runux-tnn-engine`) unifiant les deux hémisphères sous l'autorité formelle du PFC.

## Étape 2.5 : Benchmarking "Bare-Metal" et Certification Phase 2
*   **Action** : Relancer les 10 expériences physiques avec le nouveau moteur compilé.
*   **Action** : Comparer la latence, la consommation énergétique (Green AI), et l'occupation MXU par rapport à la Phase 1 (PyTorch).
*   **Livrable** : Un papier de recherche étendu (Research Paper v2.0) actant la naissance du premier *Univers Model* formellement certifié et hardware-accéléré au monde.
