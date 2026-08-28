import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path("paper_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
fig.patch.set_facecolor("#0b0f19")

for ax in axes.flat:
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#374151")

# Panel 1: Subject 1 - Fractional Chern Insulator (Chern Invariant Error)
models_s1 = ["TNN (Holonomy)", "Trad MLP", "Noise Baseline"]
err_s1 = [0.537, 0.588, 10.588]
colors_s1 = ["#34d399", "#f59e0b", "#ef4444"]
bars1 = axes[0].bar(models_s1, err_s1, color=colors_s1, width=0.55)
axes[0].set_yscale("log")
axes[0].set_title("Subject 1: Chern Invariant Error $|C - C_{\\text{exact}}|$", color="#34d399", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Error (log scale)", color="#9ca3af")
for bar in bars1:
    y = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., y * 1.15, f"{y:.3f}", ha="center", va="bottom", color="#e5e7eb", fontsize=9, fontweight="bold")

# Panel 2: Subject 2 - Relativistic GRMHD (Magnetic Divergence |div B|)
models_s2 = ["TNN (Solenoidal)", "Trad CNN", "Noise Baseline"]
div_s2 = [5.98e-10, 0.065, 0.853]
colors_s2 = ["#38bdf8", "#f59e0b", "#ef4444"]
bars2 = axes[1].bar(models_s2, div_s2, color=colors_s2, width=0.55)
axes[1].set_yscale("log")
axes[1].set_title("Subject 2: Solenoidal Violation $\\Vert\\nabla \\cdot \\mathbf{B}\\Vert$", color="#38bdf8", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Violation (log scale)", color="#9ca3af")
axes[1].text(0, 2e-9, "Exact 0.000\n($1.1\\times 10^8\\text{x}$ suppression)", ha="center", color="#38bdf8", fontweight="bold", fontsize=9)

# Panel 3: Subject 3 - Epigenetic Chromatin (2nd Law Thermodynamics Violation)
models_s3 = ["TNN (Onsager)", "Trad MLP", "Noise Baseline"]
viol_s3 = [0.000, 0.041, 0.512]
colors_s3 = ["#a78bfa", "#f59e0b", "#ef4444"]
bars3 = axes[2].bar(models_s3, viol_s3, color=colors_s3, width=0.55)
axes[2].set_title("Subject 3: 2nd Law Violation ($\\dot{S} < 0$ Fraction)", color="#a78bfa", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Violation Rate", color="#9ca3af")
axes[2].set_ylim(-0.02, 0.6)
for bar in bars3:
    y = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., y + 0.02, f"{y:.3f}", ha="center", va="bottom", color="#e5e7eb", fontsize=9, fontweight="bold")

plt.tight_layout()
out_png = FIG_DIR / "blind_inference_triple_benchmark.png"
plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved blind benchmark figure to:", str(out_png))
