/-
================================================================================
  TopoStability.lean
  Stabilité de la filtration de Vietoris–Rips.
  AUDIT : aucune déclaration `axiom`, aucun `sorry`, aucun `admit`.
================================================================================
-/

import Mathlib

set_option autoImplicit false

namespace HoloAlg.TDA

variable {V : Type*} [DecidableEq V]

/-! ### §1. MATRICES DE DISSIMILARITÉ ET PROXIMITÉ EN NORME SUP -/

abbrev Dissim (V : Type*) : Type _ := V → V → ℝ

def Close (ε : ℝ) (d e : Dissim V) : Prop :=
  ∀ a b : V, |d a b - e a b| ≤ ε

theorem Close.symm {ε : ℝ} {d e : Dissim V} (h : Close ε d e) : Close ε e d := by
  intro a b
  rw [abs_sub_comm]
  exact h a b

theorem Close.mono {ε ε' : ℝ} {d e : Dissim V} (hle : ε ≤ ε') (h : Close ε d e) :
    Close ε' d e :=
  fun a b => le_trans (h a b) hle

theorem Close.trans {ε δ : ℝ} {d e f : Dissim V}
    (h₁ : Close ε d e) (h₂ : Close δ e f) : Close (ε + δ) d f := by
  intro a b
  have hstep : |d a b - f a b| ≤ |d a b - e a b| + |e a b - f a b| :=
    abs_sub_le (d a b) (e a b) (f a b)
  have := h₁ a b
  have := h₂ a b
  linarith

theorem Close.refl (d : Dissim V) : Close 0 d d := by
  intro a b
  simp

/-! ### §2. LE COMPLEXE DE VIETORIS–RIPS COMME FILTRATION -/

def RipsSimplex (d : Dissim V) (t : ℝ) (σ : Finset V) : Prop :=
  ∀ a ∈ σ, ∀ b ∈ σ, d a b ≤ t

theorem ripsSimplex_of_subset (d : Dissim V) (t : ℝ) {σ τ : Finset V}
    (hsub : τ ⊆ σ) (h : RipsSimplex d t σ) : RipsSimplex d t τ :=
  fun a ha b hb => h a (hsub ha) b (hsub hb)

theorem ripsSimplex_mono (d : Dissim V) {t t' : ℝ} (hle : t ≤ t') {σ : Finset V}
    (h : RipsSimplex d t σ) : RipsSimplex d t' σ :=
  fun a ha b hb => le_trans (h a ha b hb) hle

theorem ripsSimplex_empty (d : Dissim V) (t : ℝ) : RipsSimplex d t (∅ : Finset V) := by
  intro a ha
  simp at ha

/-! ### §3. ENTRELACEMENT — RÉSULTAT CENTRAL -/

theorem rips_shift {ε : ℝ} {d e : Dissim V} (hde : Close ε d e) (t : ℝ)
    {σ : Finset V} (h : RipsSimplex d t σ) : RipsSimplex e (t + ε) σ := by
  intro a ha b hb
  have habs := abs_le.mp (hde a b)
  have hd := h a ha b hb
  linarith [habs.1, habs.2]

theorem rips_interleaving {ε : ℝ} {d e : Dissim V} (hde : Close ε d e) (t : ℝ)
    (σ : Finset V) :
    (RipsSimplex d t σ → RipsSimplex e (t + ε) σ) ∧
    (RipsSimplex e t σ → RipsSimplex d (t + ε) σ) :=
  ⟨rips_shift hde t, rips_shift hde.symm t⟩

theorem rips_interleaving_roundtrip {ε : ℝ} {d e : Dissim V} (hde : Close ε d e)
    (t : ℝ) {σ : Finset V} (h : RipsSimplex d t σ) :
    RipsSimplex d (t + ε + ε) σ :=
  rips_shift hde.symm (t + ε) (rips_shift hde t h)

/-! ### §4. VALEUR DE FILTRATION ET CARACTÈRE 1-LIPSCHITZIEN -/

theorem prod_nonempty {σ : Finset V} (hσ : σ.Nonempty) : (σ ×ˢ σ).Nonempty := by
  obtain ⟨a, ha⟩ := hσ
  exact ⟨(a, a), Finset.mem_product.mpr ⟨ha, ha⟩⟩

noncomputable def ripsValue (d : Dissim V) (σ : Finset V) (hσ : σ.Nonempty) : ℝ :=
  (σ ×ˢ σ).sup' (prod_nonempty hσ) (fun p => d p.1 p.2)

theorem ripsValue_le_iff (d : Dissim V) {σ : Finset V} (hσ : σ.Nonempty) (t : ℝ) :
    ripsValue d σ hσ ≤ t ↔ RipsSimplex d t σ := by
  constructor
  · intro hle a ha b hb
    have h_mem : (a, b) ∈ σ ×ˢ σ := Finset.mem_product.mpr ⟨ha, hb⟩
    have key : d a b ≤ ripsValue d σ hσ := by
      unfold ripsValue
      rw [Finset.le_sup'_iff]
      exact ⟨(a, b), h_mem, le_rfl⟩
    exact le_trans key hle
  · intro hσt
    apply Finset.sup'_le
    rintro ⟨a, b⟩ hab
    rcases Finset.mem_product.mp hab with ⟨ha, hb⟩
    exact hσt a ha b hb

theorem ripsSimplex_ripsValue (d : Dissim V) {σ : Finset V} (hσ : σ.Nonempty) :
    RipsSimplex d (ripsValue d σ hσ) σ :=
  (ripsValue_le_iff d hσ _).mp le_rfl

theorem ripsValue_lipschitz {ε : ℝ} {d e : Dissim V} (hde : Close ε d e)
    {σ : Finset V} (hσ : σ.Nonempty) :
    |ripsValue d σ hσ - ripsValue e σ hσ| ≤ ε := by
  rw [abs_sub_le_iff]
  constructor
  · have h1 : RipsSimplex e (ripsValue e σ hσ) σ := ripsSimplex_ripsValue e hσ
    have h2 : RipsSimplex d (ripsValue e σ hσ + ε) σ := rips_shift hde.symm _ h1
    have h3 : ripsValue d σ hσ ≤ ripsValue e σ hσ + ε :=
      (ripsValue_le_iff d hσ _).mpr h2
    linarith
  · have h1 : RipsSimplex d (ripsValue d σ hσ) σ := ripsSimplex_ripsValue d hσ
    have h2 : RipsSimplex e (ripsValue d σ hσ + ε) σ := rips_shift hde _ h1
    have h3 : ripsValue e σ hσ ≤ ripsValue d σ hσ + ε :=
      (ripsValue_le_iff e hσ _).mpr h2
    linarith

/-! ### §5. TRANSFORMATION CORRÉLATION → DISTANCE -/

def corrToDist (r : ℝ) : ℝ := 1 - r

theorem corrToDist_isometry (r s : ℝ) : |corrToDist r - corrToDist s| = |r - s| := by
  unfold corrToDist
  rw [show (1 - r) - (1 - s) = -(r - s) by ring, abs_neg]

theorem corrToDist_close {δ : ℝ} {R S : Dissim V}
    (h : ∀ a b : V, |R a b - S a b| ≤ δ) :
    Close δ (fun a b => corrToDist (R a b)) (fun a b => corrToDist (S a b)) := by
  intro a b
  show |corrToDist (R a b) - corrToDist (S a b)| ≤ δ
  rw [corrToDist_isometry]
  exact h a b

theorem pipeline_stability {δ : ℝ} {R S : Dissim V}
    (h : ∀ a b : V, |R a b - S a b| ≤ δ)
    {σ : Finset V} (hσ : σ.Nonempty) :
    |ripsValue (fun a b => corrToDist (R a b)) σ hσ
     - ripsValue (fun a b => corrToDist (S a b)) σ hσ| ≤ δ :=
  ripsValue_lipschitz (corrToDist_close h) hσ

#print axioms ripsValue_lipschitz

end HoloAlg.TDA
