import Lake
open Lake DSL

package HoloEngine where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib HoloEngine where
  srcDir := "."
  roots := #[`HoloEngine]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
