# Best Practices and Structural Hardness

This document defines the mathematical rigor, engineering constraints, and epistemological rules (Hardness) required for developing the **TNN (Thermodynamic, Topological, Tensor Neural Network) Univers Model**.

## 1. Structural Hardness & Rigor
To ensure the Univers Model "calculates" physics rather than approximating it, all implementations must adhere to the following absolute invariants:

*   **Zero-Sorry Verification**: All core algebraic rules (e.g., T-Dual bounce, Aubin-Lions compactness) must be strictly formalized and machine-verified using Lean 4. There must be zero gaps in the pre-geometric proofs.
*   **Symplectic Conservation**: The Predictor block (Neural SDE) must strictly conserve energy. The sum of kinetic and potential energy must remain constant over long rollouts. The Predictor must traverse the path of least action.
*   **Exact Equivariance**: The Topo-Encoder must guarantee exact $SE(3)$ or $E(n)$ invariance. Rotating, translating, or permuting the physical input must result in the mathematically identical latent state or a predictably transformed equivariant tensor.
*   **Strict Rulial Inversions**: Singularity prevention (like the black hole horizon or Navier-Stokes cascade collapse) must not rely on artificial mathematical cutoffs, but on exact Rulial Inversions (e.g., $R_{\text{eff}} = \max(r, \alpha'/r)$) flipping $N$-arity structures to inverted dual bonds.

## 2. Poly-Algebraic Best Practices
*   **Use Irreducible Arities**: Represent atomic, hadronic, and macroscopic systems using $\Xi^{\langle N \rangle}$. Do not decompose a 3-arity color-singlet (proton) into three 2-arity gluon bonds.
*   **Virtual Heat Tracking**: Standard CPUs/GPUs struggle with $N$-arity topology. In the vHPU emulator, actively track and document "Virtual Heat" (clock cycle waste) to justify the eventual hardware migration to physical acoustic-fluidic architectures.
*   **Energy Critic over Statistical Loss**: Do not use Mean Squared Error (MSE) to train the model. The loss must be an Energy Critic that calculates the thermodynamic impossibility of a state transition (e.g., penalizing localized entropy decreases without external work).

## 3. Integration Guidelines (The Chimera Architecture)
When assembling the engine, follow the predefined pipeline:
1.  **Data Layer**: NVIDIA Modulus for massive GPU-optimized physical data handling.
2.  **Encoder**: `e3nn` (Euclidean Neural Networks) to map coordinates into invariant tensors.
3.  **Predictor**: `lagrangian_nns` (Cranmer et al.) for calculating the latent future state.
4.  **Orchestrator**: V-JEPA (Joint Embedding Predictive Architecture) loop with masking to enforce self-supervised physical learning.
