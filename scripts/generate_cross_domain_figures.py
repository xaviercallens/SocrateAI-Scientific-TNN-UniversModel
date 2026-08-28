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

use_cases = [
    "Astro-Plasma Jet",
    "Induced Seismicity",
    "Ocean-Atmos FNO",
    "Tumor Mechanobio",
    "Supercon Metamat"
]
x_pos = np.arange(len(use_cases))

# Plot 1: Accuracy Gain Factor (TNN vs Baseline)
acc_gains = [467.8, 285.9, 467.1, 372.4, 242.9]
bars1 = axes[0].bar(x_pos, acc_gains, color="#38bdf8", alpha=0.9, width=0.55)
axes[0].set_title("Accuracy Gain (x-fold improvement over Baseline)", color="#38bdf8", fontsize=11, fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(use_cases, rotation=25, ha="right", color="#e5e7eb")
axes[0].set_ylabel("Gain Multiplier (x)", color="#9ca3af")
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.0f}x", ha="center", va="bottom", color="#38bdf8", fontweight="bold", fontsize=9)

# Plot 2: Topological Betti-1 Persistence Preservation (%)
base_b1 = [38, 45, 52, 41, 48]
tnn_b1 = [100, 98, 100, 99, 100]
width = 0.35
axes[1].bar(x_pos - width/2, base_b1, width=width, label="Baseline (MLP/CNN)", color="#ef4444", alpha=0.85)
axes[1].bar(x_pos + width/2, tnn_b1, width=width, label="TNN / TDA Operator", color="#34d399", alpha=0.9)
axes[1].set_title("Topological Invariant Preservation ($H_1$ Loop %)", color="#34d399", fontsize=11, fontweight="bold")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(use_cases, rotation=25, ha="right", color="#e5e7eb")
axes[1].set_ylabel("Preservation (%)", color="#9ca3af")
axes[1].set_ylim(0, 115)
axes[1].legend(facecolor="#1e293b", edgecolor="#374151", labelcolor="#e5e7eb", loc="upper left")

# Plot 3: Inference Latency / Speedup Profile (ms)
lat_base = [635.6, 36.1, 220.3, 28.1, 42.1]
lat_tnn = [1969.2, 18.1, 0.86, 12.2, 18.3]
axes[2].plot(x_pos, lat_base, "o--", color="#ef4444", label="Baseline (ms)", linewidth=2, markersize=7)
axes[2].plot(x_pos, lat_tnn, "s-", color="#818cf8", label="TNN (ms)", linewidth=2, markersize=7)
axes[2].set_yscale("log")
axes[2].set_title("Inference Latency (ms, log-scale)", color="#818cf8", fontsize=11, fontweight="bold")
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(use_cases, rotation=25, ha="right", color="#e5e7eb")
axes[2].set_ylabel("Latency (ms)", color="#9ca3af")
axes[2].legend(facecolor="#1e293b", edgecolor="#374151", labelcolor="#e5e7eb")

plt.tight_layout()
out_png = FIG_DIR / "cross_domain_generalization_benchmark.png"
plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved figure successfully to:", str(out_png))
