#!/usr/bin/env python3
"""
Scientific Visualization Suite for 10 Biomedical TDA & TNN Domains
Generates publication-quality figures showcasing Betti numbers, Waddington landscapes,
Hi-C loops, and thermodynamic Onsager convergence.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "./paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = "/mnt/disks/disk-socrateai-local-1/bio_datasets"

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# 1. Hi-C 3D Chromatin Loops
hic_coords = np.load(f"{DATA_DIR}/01_hic_chromatin_coords.npy")
ax = axes[0, 0]
ax.plot(hic_coords[:, 0], hic_coords[:, 1], color='#1f77b4', lw=1.5, alpha=0.8)
ax.scatter(hic_coords[50:150, 0], hic_coords[50:150, 1], color='#d62728', s=15, label='TAD 1 (CTCF Loop)')
ax.scatter(hic_coords[200:320, 0], hic_coords[200:320, 1], color='#2ca02c', s=15, label='TAD 2 (Cohesin Anchor)')
ax.set_title("1. 3D Chromatin Polymer Loops (Hi-C GSE63525)", fontsize=11, fontweight='bold')
ax.set_xlabel("X (Spatial Scale)")
ax.set_ylabel("Y (Spatial Scale)")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 2. scRNA Waddington Potential Landscape
scrna_lat = np.load(f"{DATA_DIR}/02_scrna_latent_manifold.npy")
ax = axes[0, 1]
scatter = ax.scatter(scrna_lat[:, 0], scrna_lat[:, 1], c=scrna_lat[:, 2], cmap='viridis', s=8, alpha=0.7)
ax.set_title("2. Cancer Waddington Differentiation (scRNA)", fontsize=11, fontweight='bold')
ax.set_xlabel("Pseudotime Progression")
ax.set_ylabel("Clonal Branching (Clone A vs B)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Waddington Potential $\\Psi(x)$")
ax.grid(True, alpha=0.3)

# 3. Spatial Visium Tumor Infiltration
visium_coords = np.load(f"{DATA_DIR}/04_spatial_visium_coords.npy")
visium_prof = np.load(f"{DATA_DIR}/04_spatial_visium_profiles.npy")
ax = axes[0, 2]
scatter = ax.scatter(visium_coords[:, 0], visium_coords[:, 1], c=visium_prof[:, 1], cmap='plasma', s=10)
ax.set_title("3. Spatial Transcriptomics (10x Visium Infiltration)", fontsize=11, fontweight='bold')
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Immune Infiltration Density")
ax.grid(True, alpha=0.3)

# 4. DNA Methylome Pan-Cancer Separation
cpg_beta = np.load(f"{DATA_DIR}/05_dna_methylation_beta.npy")
cpg_labels = np.load(f"{DATA_DIR}/05_cancer_labels.npy")
u, s, vt = np.linalg.svd(cpg_beta - np.mean(cpg_beta, axis=0), full_matrices=False)
cpg_pca = np.dot(cpg_beta, vt[:2].T)
ax = axes[1, 0]
ax.scatter(cpg_pca[cpg_labels==0, 0], cpg_pca[cpg_labels==0, 1], color='#2ca02c', alpha=0.7, s=20, label='Normal CpG Baseline')
ax.scatter(cpg_pca[cpg_labels==1, 0], cpg_pca[cpg_labels==1, 1], color='#d62728', alpha=0.7, s=20, label='Cancer Hyper/Hypo Methylome')
ax.set_title("4. Pan-Cancer CpG Epigenetic Topologies", fontsize=11, fontweight='bold')
ax.set_xlabel("Topological Mode 1")
ax.set_ylabel("Topological Mode 2")
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# 5. TDA Betti Numbers Across 10 Domains
with open("./tda_biomedical_results/tda_biomedical_10_domains_audit.json") as f:
    tda_data = json.load(f)
domain_names = [k.replace("0", "").replace("_", " ")[:14] for k in tda_data.keys()]
b0_vals = [v['betti_0_clusters'] for v in tda_data.values()]
b1_vals = [v['betti_1_loops'] for v in tda_data.values()]
ax = axes[1, 1]
x_pos = np.arange(len(domain_names))
ax.bar(x_pos - 0.2, b0_vals, width=0.4, label='$\\beta_0$ (Clusters/Components)', color='#1f77b4')
ax.bar(x_pos + 0.2, b1_vals, width=0.4, label='$\\beta_1$ (Loops/Vortices)', color='#ff7f0e')
ax.set_xticks(x_pos)
ax.set_xticklabels(domain_names, rotation=45, ha='right', fontsize=8)
ax.set_title("5. TDA Betti Invariants Across 10 Domains", fontsize=11, fontweight='bold')
ax.set_ylabel("Persistent Homology Count")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 6. TNN Loss Reduction & Convergence Factor
with open("./certs/biomedical_10_domains_certification.json") as f:
    cert_data = json.load(f)
domains_tnn = [k.replace("domain_", "").replace("_", " ")[:14] for k in cert_data.keys()]
reduction_factors = [np.log10(v['loss_reduction_factor'] + 1.0) for v in cert_data.values()]
ax = axes[1, 2]
bars = ax.barh(np.arange(len(domains_tnn)), reduction_factors, color='#9467bd', alpha=0.85)
ax.set_yticks(np.arange(len(domains_tnn)))
ax.set_yticklabels(domains_tnn, fontsize=8)
ax.set_title("6. TNN Convergence ($log_{10}$ Loss Reduction)", fontsize=11, fontweight='bold')
ax.set_xlabel("$log_{10}(\\mathcal{L}_{\\mathrm{init}} / \\mathcal{L}_{\\mathrm{final}})$")
ax.grid(True, alpha=0.3)

out_file = f"{OUTPUT_DIR}/biomedical_10_domains_tda_tnn.png"
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Multi-Domain Scientific Visualization saved to: {out_file}")
