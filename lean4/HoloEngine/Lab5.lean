import Mathlib

set_option autoImplicit false

namespace HoloAlg.Lab5

variable {X : Type*}

abbrev ScalarField (X : Type*) := X → ℝ

def MaxNormClose (ε : ℝ) (f g : ScalarField X) : Prop :=
  ∀ x : X, |f x - g x| ≤ ε

theorem MaxNormClose.symm {ε : ℝ} {f g : ScalarField X} (h : MaxNormClose ε f g) : 
    MaxNormClose ε g f := by
  intro x
  rw [abs_sub_comm]
  exact h x

theorem MaxNormClose.trans {ε δ : ℝ} {f g h_field : ScalarField X}
    (h1 : MaxNormClose ε f g) (h2 : MaxNormClose δ g h_field) : 
    MaxNormClose (ε + δ) f h_field := by
  intro x
  have step := abs_sub_le (f x) (g x) (h_field x)
  have e1 := h1 x
  have e2 := h2 x
  linarith

def Sublevel (f : ScalarField X) (t : ℝ) : Set X :=
  {x : X | f x ≤ t}

theorem sublevel_mono (f : ScalarField X) {t1 t2 : ℝ} (hle : t1 ≤ t2) :
    Sublevel f t1 ⊆ Sublevel f t2 := by
  intro x hx
  exact le_trans hx hle

theorem sublevel_interleaving {ε : ℝ} {f g : ScalarField X} (h : MaxNormClose ε f g) (t : ℝ) :
    Sublevel f t ⊆ Sublevel g (t + ε) := by
  intro x hx
  have h_abs := abs_le.mp (h x)
  have step : g x - f x ≤ ε := by
    have h1 := h_abs.1
    have h2 := h_abs.2
    linarith
  have hx_le : f x ≤ t := hx
  exact show g x ≤ t + ε by linarith

variable (α' : ℝ) (hα : 0 < α')

noncomputable def P4_Reff (α' R : ℝ) : ℝ := max R (α' / R)

noncomputable def P4_Field (α' : ℝ) (f : ScalarField X) : ScalarField X :=
  fun x => P4_Reff α' (f x)

theorem p4_topological_void (α' : ℝ) (hα : 0 < α') (f : ScalarField X) (hf : ∀ x, 0 < f x) (t : ℝ) (ht : t < Real.sqrt α') :
    Sublevel (P4_Field α' f) t = ∅ := by
  ext x
  dsimp [Sublevel]
  simp only [Set.mem_empty_iff_false, iff_false, not_le]
  have h_bound : Real.sqrt α' ≤ P4_Field α' f x := by
    unfold P4_Field P4_Reff
    by_cases h1 : Real.sqrt α' ≤ f x
    · exact le_max_of_le_left h1
    · apply le_max_of_le_right
      have h2 : f x < Real.sqrt α' := not_le.mp h1
      have hs : (0 : ℝ) ≤ Real.sqrt α' := Real.sqrt_nonneg α'
      have key : Real.sqrt α' * f x ≤ Real.sqrt α' * Real.sqrt α' :=
        mul_le_mul_of_nonneg_left h2.le hs
      rw [le_div_iff₀ (hf x)]
      rwa [Real.mul_self_sqrt hα.le] at key
  linarith

theorem p4_macroscopic_isometry (α' : ℝ) (hα : 0 < α') {ε : ℝ} {f g : ScalarField X} 
    (h_macro_f : ∀ x, Real.sqrt α' ≤ f x)
    (h_macro_g : ∀ x, Real.sqrt α' ≤ g x)
    (h_close : MaxNormClose ε f g) : 
    MaxNormClose ε (P4_Field α' f) (P4_Field α' g) := by
  intro x
  have hf_eq : P4_Field α' f x = f x := by
    unfold P4_Field P4_Reff
    apply max_eq_left
    rw [div_le_iff₀ (lt_of_lt_of_le (Real.sqrt_pos.mpr hα) (h_macro_f x))]
    have sq_le : Real.sqrt α' * Real.sqrt α' ≤ f x * f x :=
      mul_le_mul (h_macro_f x) (h_macro_f x) (Real.sqrt_nonneg α') (lt_of_lt_of_le (Real.sqrt_pos.mpr hα) (h_macro_f x)).le
    rwa [Real.mul_self_sqrt hα.le] at sq_le
    
  have hg_eq : P4_Field α' g x = g x := by
    unfold P4_Field P4_Reff
    apply max_eq_left
    rw [div_le_iff₀ (lt_of_lt_of_le (Real.sqrt_pos.mpr hα) (h_macro_g x))]
    have sq_le : Real.sqrt α' * Real.sqrt α' ≤ g x * g x :=
      mul_le_mul (h_macro_g x) (h_macro_g x) (Real.sqrt_nonneg α') (lt_of_lt_of_le (Real.sqrt_pos.mpr hα) (h_macro_g x)).le
    rwa [Real.mul_self_sqrt hα.le] at sq_le

  rw [hf_eq, hg_eq]
  exact h_close x

end HoloAlg.Lab5
