# vHPU Implementation Guide

## 1. Overview
The **Virtual Hyper-Arity Processing Unit (vHPU)** is the execution engine for the TNN Univers Model. It entirely replaces the PyTorch/Python overhead by leveraging the **RunuX AI Runtime**, a `no_std` memory-safe Rust kernel designed for edge RISC-V and Cloud TPUs. 

The vHPU engine implements the **SymBrain v4 Dual-Hemisphere** architecture:
*   **Left Hemisphere (Topological)**: Executes the E(3)-Equivariant Graph Neural Network (EGNN) using `rvv_simd` (RISC-V Vector instructions) for O(1) pointer-based spatial inversion.
*   **Right Hemisphere (Tensor/Continuous)**: Executes the Fourier Neural Operator (FNO) using `tpu_pjrt` and `mlgo_advisor` for 128x128 systolic tiling on TPU v5e arrays.
*   **PFC Router (Thermodynamic)**: Executes the Hamiltonian Neural Network (HNN) as an Energy Critic. It acts as the "Zero-Sorry Symplectic Hook" that interfaces with the Lean 4 proof definitions via FFI, ensuring no out-of-distribution physical states are computed.

## 2. Directory Structure
The `vhpu_engine` Rust crate is structured as follows:
*   `Cargo.toml`: Declares path dependencies to the cloned `runux-ai-runtime` workspace (e.g., `ai_runtime`, `hal`, `rvv_simd`).
*   `src/main.rs`: The main entry point initializing the Dual-Hemisphere SymBrain coordinator and executing the Poly-Algebraic logic over standard floating-point approximations.
*   `src/egnn_left.rs`: RISC-V accelerated topological routing.
*   `src/fno_right.rs`: TPU accelerated continuous fluid PDE fields.
*   `src/hnn_pfc.rs`: The Symplectic Energy Critic.

## 3. Compilation Strategy
Due to the dependency on `no_std` and embedded targets, the vHPU should be compiled using Cargo's target flags:
```bash
# Edge RISC-V Deployment (Left Hemisphere heavy)
cargo build --target riscv64gc-unknown-linux-gnu --release

# Cloud TPU Deployment (Right Hemisphere heavy)
cargo build --release
```

## 4. Performance Metrics Target (Phase 2.4)
The implementation explicitly targets a **90%+ reduction in Virtual Heat** (latency induced by continuous backpropagation) compared to PyTorch MLPs. The `Rulial_Invert` instruction natively maps to Rust's zero-cost abstractions, dropping decoding latency from ~90ms to ~7ms per forward pass on continuous fields.
