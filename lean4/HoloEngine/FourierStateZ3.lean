/-
  MechanicaFluidorum/FourierStateZ3.lean — Tier A
  ========================================================================
  Formalizes the state space representation for the 3D Fourier-Galerkin
  incompressible Navier-Stokes system.
-/
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset

namespace MechanicaFluidorum.FourierZ3

open Complex
open scoped BigOperators

/-! ### 1. The Wavevector Lattice -/

def Wavevector := Fin 3 → ℤ

def k_sq (k : Wavevector) : ℤ :=
  ∑ i : Fin 3, (k i) ^ 2

def zero_mode : Wavevector := fun _ => 0

def negWavevector (k : Wavevector) : Wavevector := fun i => - k i

/-! ### 2. Complex Dot Product & Transversality -/

noncomputable def fourier_dot (k : Wavevector) (v : Fin 3 → ℂ) : ℂ :=
  ∑ i : Fin 3, (k i : ℂ) * v i

/-! ### 3. The Constrained Kinematic State Space -/

structure FourierState where
  u : Wavevector → (Fin 3 → ℂ)
  div_free : ∀ k : Wavevector, fourier_dot k (u k) = 0
  conj_sym : ∀ k : Wavevector, ∀ i : Fin 3, u (negWavevector k) i = star (u k i)
  zero_mean : ∀ i : Fin 3, u zero_mode i = 0

theorem zero_is_div_free : ∀ k : Wavevector, fourier_dot k (fun _ => (0 : ℂ)) = 0 := by
  intro k
  unfold fourier_dot
  exact Finset.sum_eq_zero (fun i _ => mul_zero _)

noncomputable def zero_FourierState : FourierState := {
  u := fun _ _ => 0,
  div_free := zero_is_div_free,
  conj_sym := fun _ _ => by simp,
  zero_mean := fun _ => rfl
}

/-! ### 4. The Leray-Hopf Projector (Physics Enforcement) -/

noncomputable def apply_leray (k : Wavevector) (v : Fin 3 → ℂ) : Fin 3 → ℂ :=
  if hk : k_sq k = 0 then
    v
  else
    let k_dot_v := fourier_dot k v
    let k_sq_c : ℂ := (k_sq k : ℂ)
    fun i => v i - (k_dot_v / k_sq_c) * (k i : ℂ)

theorem leray_projector_divergence_free (k : Wavevector) (v : Fin 3 → ℂ) (hk : k_sq k ≠ 0) :
    fourier_dot k (apply_leray k v) = 0 := by
  unfold apply_leray
  simp only [hk, dite_false]
  unfold fourier_dot
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]

  have H : (∑ i : Fin 3, (k i : ℂ) * ((∑ j : Fin 3, (k j : ℂ) * v j) / (k_sq k : ℂ) * (k i : ℂ))) =
      ((∑ j : Fin 3, (k j : ℂ) * v j) / (k_sq k : ℂ)) * (k_sq k : ℂ) := by
    unfold k_sq
    push_cast
    calc
      (∑ i : Fin 3, (k i : ℂ) * ((∑ j : Fin 3, (k j : ℂ) * v j) / (∑ j : Fin 3, (k j : ℂ) ^ 2) * (k i : ℂ)))
        = ∑ i : Fin 3, ((∑ j : Fin 3, (k j : ℂ) * v j) / (∑ j : Fin 3, (k j : ℂ) ^ 2)) * ((k i : ℂ) ^ 2) := by
          apply Finset.sum_congr rfl
          intro x _
          ring
      _ = ((∑ j : Fin 3, (k j : ℂ) * v j) / (∑ j : Fin 3, (k j : ℂ) ^ 2)) * ∑ i : Fin 3, ((k i : ℂ) ^ 2) := by
          rw [← Finset.mul_sum]
  rw [H]

  have hk_c : (k_sq k : ℂ) ≠ 0 := by exact_mod_cast hk
  rw [div_mul_cancel₀ _ hk_c, sub_self]

theorem leray_projector_idempotent (k : Wavevector) (v : Fin 3 → ℂ) :
    apply_leray k (apply_leray k v) = apply_leray k v := by
  by_cases hk : k_sq k = 0
  · unfold apply_leray
    simp [hk]
  · ext i
    have hdiv := leray_projector_divergence_free k v hk
    unfold apply_leray at hdiv
    simp only [hk, dite_false] at hdiv
    unfold apply_leray
    simp only [hk, dite_false]
    simp_rw [hdiv]
    simp

#print axioms leray_projector_divergence_free

end MechanicaFluidorum.FourierZ3
