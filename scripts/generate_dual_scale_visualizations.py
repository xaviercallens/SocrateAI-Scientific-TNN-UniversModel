import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path("/mnt/disks/disk-socrateai-local-1/dual_scale_datasets")
if not DATA_DIR.exists():
    DATA_DIR = Path("./data/real/dual_scale_datasets")

FIG_DIR = Path("paper_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
fig.patch.set_facecolor("#0b0f19")

for ax in axes.flat:
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#374151")

# Panel 1: Quantum Berry Curvature (Domain 1)
d1 = np.load(DATA_DIR / "domain01_topological_materials_berry.npz")
im1 = axes[0, 0].imshow(d1["berry_z"][:, :, 12], cmap="magma", extent=[-np.pi, np.pi, -np.pi, np.pi])
axes[0, 0].set_title("Quantum Materials: Berry Curvature $\\Omega_z(\\mathbf{k})$", color="#38bdf8", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("$k_x$", color="#9ca3af")
axes[0, 0].set_ylabel("$k_y$", color="#9ca3af")
cbar1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
cbar1.ax.tick_params(colors="#9ca3af")

# Panel 2: Plasma Tokamak Drift-Wave Potential (Domain 2)
d2 = np.load(DATA_DIR / "domain02_plasma_tokamak_gyrokinetics.npz")
im2 = axes[0, 1].imshow(d2["potential"][:, :, 16], cmap="plasma", aspect="auto")
axes[0, 1].set_title("Tokamak Fusion: Gyrokinetic $\\delta\\phi(r,\\theta)$", color="#38bdf8", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("$\\theta$ (Poloidal)", color="#9ca3af")
axes[0, 1].set_ylabel("$r$ (Radial)", color="#9ca3af")
cbar2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors="#9ca3af")

# Panel 3: HEA Dislocation Lines (Domain 3)
d3 = np.load(DATA_DIR / "domain03_hea_dislocations_cantor.npz")
nodes = d3["nodes"]
axes[0, 2].scatter(nodes[:, 0], nodes[:, 1], c=nodes[:, 2], cmap="viridis", s=18, alpha=0.85)
axes[0, 2].set_title("High-Entropy Alloy: Frank-Read Dislocation Loop", color="#38bdf8", fontsize=11, fontweight="bold")
axes[0, 2].set_xlabel("X (nm)", color="#9ca3af")
axes[0, 2].set_ylabel("Y (nm)", color="#9ca3af")

# Panel 4: Connectome Cortical BOLD Signals (Domain 4)
d4 = np.load(DATA_DIR / "domain04_hcp_connectome_neurodynamics.npz")
bold = d4["bold"]
for i in range(min(8, len(bold))):
    axes[1, 0].plot(bold[i, :100], alpha=0.75, linewidth=1.2)
axes[1, 0].set_title("Neurodynamics: Multi-Region BOLD Oscillations", color="#38bdf8", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Time (TR)", color="#9ca3af")
axes[1, 0].set_ylabel("Amplitude", color="#9ca3af")

# Panel 5: CloudSat Radar Reflectivity Curtain (Domain 5)
d5 = np.load(DATA_DIR / "domain05_cloudsat_radar_microphysics.npz")
im5 = axes[1, 1].imshow(d5["reflectivity_dbz"].T, cmap="turbo", aspect="auto", origin="lower",
                        extent=[-60, 60, 0, 18])
axes[1, 1].set_title("Cloud Microphysics: CloudSat Radar Reflectivity (dBZ)", color="#38bdf8", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Latitude ($^\\circ$)", color="#9ca3af")
axes[1, 1].set_ylabel("Altitude (km)", color="#9ca3af")
cbar5 = fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar5.ax.tick_params(colors="#9ca3af")

# Panel 6: Porous Media Supercritical CO2 (Domain 9)
d9 = np.load(DATA_DIR / "domain09_digitalrocks_berea_co2.npz")
im6 = axes[1, 2].imshow(d9["co2_sat"][:, :, 16], cmap="inferno")
axes[1, 2].set_title("Porous Media: Supercritical $\\text{CO}_2$ Saturation", color="#38bdf8", fontsize=11, fontweight="bold")
axes[1, 2].set_xlabel("X ($3D\\ \\mu$-CT)", color="#9ca3af")
axes[1, 2].set_ylabel("Y", color="#9ca3af")
cbar6 = fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
cbar6.ax.tick_params(colors="#9ca3af")

plt.tight_layout()
out_png = FIG_DIR / "dual_scale_10_domains_tda_tnn.png"
plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()

print("Generated visual panel successfully at:", str(out_png))
