# Roadmap: TODO, NOTODO, and Frontier

This document outlines the actionable roadmap for the SocrateAI TNN Univers Model, including strict anti-patterns (NOTODOs) and the long-term vision (Frontier).

## ✅ TODO (Immediate Action Items)
1.  **Infrastructure Setup**: Install and configure **NVIDIA Modulus** as the base framework for data ingestion and tensor operations.
2.  **Topo-Encoder Implementation**: Integrate `e3nn` and `vgsatorras/egnn`. Write the translation layer that converts pre-geometric hypergraphs into equivariant latent representations.
3.  **Thermodynamic Predictor Integration**: Fork and adapt `MilesCranmer/lagrangian_nns` to serve as the core time-stepping predictor.
4.  **V-JEPA Loop Construction**: Implement the Meta V-JEPA architecture. Build the Energy Critic to evaluate the physical validity of latent state transitions via thermodynamic principles.
5.  **vHPU Dataset Ingestion**: Feed the first macroscopic dataset into the Python vHPU emulator. (Decision needed: Johns Hopkins Turbulence Database 3D tensor vs. simplified 2D toy model).
6.  **Lean 4 Autoformalization**: Begin porting the Python Poly-Algebraic syntax into Lean 4 for formal mathematical verification.

## 🚫 NOTODO (Strict Anti-Patterns)
1.  **DO NOT use LLMs or standard Transformers**: They predict symbolic sequence probability, not physical causality. They have no innate understanding of energy or geometry.
2.  **DO NOT use Mean-Squared Error (MSE) loss**: Physical systems are not statistical averages. State prediction must be bounded by the Energy Critic (Lagrangian action and entropy).
3.  **DO NOT fall into the Binary Trap**: Never decompose an irreducible $N$-arity interaction (like a 3-quark proton) into pairwise 2-body relationships ($\sum_{i<j} V(i,j)$).
4.  **DO NOT assume continuous spacetime**: Do not start with a rigid $\mathbb{R}^d$ coordinate manifold. Space must emerge from hypergraph contractions.
5.  **DO NOT hardcode mathematical cutoffs**: Singularities should not be avoided by throwing a `max(epsilon, x)` into a formula. They must be resolved via structural Rulial Inversions (e.g., T-Dual bounce).

## 🌌 The Frontier (Long-Term Vision)
1.  **Physical Hardware Instantiation**: Transitioning the vHPU from a Python emulator (which burns "Virtual Heat") to a physical computational medium. Prototypes include Macroscopic Braid Boards (stepper motors crossing fibers) or Acoustic-Fluidic tables.
2.  **Operadic Quantum Chromodynamics**: Developing operadic representations of $N$-arity contractions for automated, high-fidelity QCD calculations and color confinement modeling.
3.  **Multi-Scale Biological Modeling**: Applying Poly-Algebraic Rulial Solvers to scRNA-seq topological networks to model multi-gene epigenetic regulation and cascade failures in oncology.
4.  **The Final Univers Model**: A unified foundation model pre-trained on celestial mechanics, fluid dynamics, and molecular behavior, capable of zero-shot generation of viable physical systems across all scales.
