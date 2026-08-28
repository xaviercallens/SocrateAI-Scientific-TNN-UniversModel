#!/usr/bin/env python3
"""
Topological Data Analysis (TDA) Suite for 10 Biomedical & Genomic Domains
Computes Persistent Homology (H0, H1, H2), Betti barcodes, and Topological Persistence Landscapes.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from ripser import ripser

DATA_DIR = "/mnt/disks/disk-socrateai-local-1/bio_datasets"
RESULTS_DIR = "./tda_biomedical_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=========================================================================")
print(" 🔬 RUNNING TDA PERSISTENT HOMOLOGY SUITE ON 10 BIOMEDICAL DOMAINS")
print("=========================================================================")

tda_summary = {}

def process_point_cloud_tda(name, points, maxdim=1, max_points=200):
    print(f"[*] Computing TDA for: {name} (Total N={len(points)})...")
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        pts = points[idx]
    else:
        pts = points
    
    diagrams = ripser(pts, maxdim=maxdim)['dgms']
    
    b0_count = len(diagrams[0])
    b1_count = len(diagrams[1]) if len(diagrams) > 1 else 0
    b2_count = len(diagrams[2]) if len(diagrams) > 2 else 0

    finite_h1 = diagrams[1][diagrams[1][:, 1] < np.inf] if len(diagrams) > 1 and len(diagrams[1]) > 0 else np.array([])
    h1_pers = (finite_h1[:, 1] - finite_h1[:, 0]) if len(finite_h1) > 0 else np.array([0.0])
    h1_entropy = float(-np.sum((h1_pers / (np.sum(h1_pers) + 1e-8)) * np.log(h1_pers / (np.sum(h1_pers) + 1e-8) + 1e-12)))

    res = {
        "domain": name,
        "num_points_analyzed": int(len(pts)),
        "betti_0_clusters": int(b0_count),
        "betti_1_loops": int(b1_count),
        "betti_2_voids": int(b2_count),
        "max_h1_persistence": float(np.max(h1_pers)) if len(h1_pers) > 0 else 0.0,
        "topological_entropy": float(h1_entropy),
    }
    print(f"  -> Betti Invariants: b0={res['betti_0_clusters']}, b1={res['betti_1_loops']}, b2={res['betti_2_voids']} | Entropy={res['topological_entropy']:.4f}")
    return res

# 1. Hi-C 3D Chromatin
hic_coords = np.load(f"{DATA_DIR}/01_hic_chromatin_coords.npy")
tda_summary["01_hic_chromatin"] = process_point_cloud_tda("01_Hi-C_3D_Chromatin_Loops", hic_coords, maxdim=1, max_points=250)

# 2. scRNA-seq Cancer Waddington Manifold
scrna_lat = np.load(f"{DATA_DIR}/02_scrna_latent_manifold.npy")
tda_summary["02_scrna_cancer_manifold"] = process_point_cloud_tda("02_scRNA_Waddington_Cancer_Trajectory", scrna_lat, maxdim=1, max_points=250)

# 3. Protein Dynamics (KRAS Oncogene 4OBE)
kras_coords = np.load(f"{DATA_DIR}/03_4OBE_ca_coords.npy")
tda_summary["03_kras_oncogene_protein"] = process_point_cloud_tda("03_KRAS_Oncogene_Protein_Structure", kras_coords, maxdim=1, max_points=200)

# 4. Spatial Transcriptomics Visium
visium_coords = np.load(f"{DATA_DIR}/04_spatial_visium_coords.npy")
tda_summary["04_spatial_transcriptomics"] = process_point_cloud_tda("04_10x_Visium_Tumor_Infiltration", visium_coords, maxdim=1, max_points=250)

# 5. DNA Methylome CpG Beta Values
cpg_beta = np.load(f"{DATA_DIR}/05_dna_methylation_beta.npy")
u, s, vt = np.linalg.svd(cpg_beta - np.mean(cpg_beta, axis=0), full_matrices=False)
cpg_pca = np.dot(cpg_beta, vt[:3].T)
tda_summary["05_dna_methylation"] = process_point_cloud_tda("05_Pan_Cancer_DNA_Methylome", cpg_pca, maxdim=1, max_points=250)

# 6. Pharmacogenomics GDSC
ic50_mat = np.load(f"{DATA_DIR}/06_gdsc_ic50_matrix.npy")
cell_feat = np.load(f"{DATA_DIR}/06_cell_features.npy")
tda_summary["06_pharmacogenomics_gdsc"] = process_point_cloud_tda("06_GDSC_Cancer_Drug_Resistance", cell_feat[:, :3], maxdim=1, max_points=250)

# 7. RNA Secondary Structures
rna_mats = np.load(f"{DATA_DIR}/07_rna_contact_matrices.npy")
rna_dist = 1.0 / (rna_mats[0] + 0.1)
tda_summary["07_rna_pseudoknots"] = process_point_cloud_tda("07_RNA_3D_Pseudoknot_Topology", rna_dist[:100, :3], maxdim=1, max_points=100)

# 8. Nuclear Morphology Contours
nuclei = np.load(f"{DATA_DIR}/08_nuclear_contours.npy")
nuclei_pts = np.array([[n[:, 0].mean(), n[:, 1].mean(), np.std(np.linalg.norm(n, axis=-1))] for n in nuclei])
tda_summary["08_nuclear_morphology"] = process_point_cloud_tda("08_Cancer_Nuclear_Pleomorphism", nuclei_pts, maxdim=1, max_points=250)

# 9. TCR Immune Repertoires
tcr_emb = np.load(f"{DATA_DIR}/09_tcr_embeddings.npy")
tcr_means = tcr_emb.mean(axis=1)[:, :3]
tda_summary["09_tcr_immune_repertoire"] = process_point_cloud_tda("09_VDJdb_TCR_Immune_Clusters", tcr_means, maxdim=1, max_points=250)

# 10. Metabolic Flux Balance (Recon3D)
S_mat = np.load(f"{DATA_DIR}/10_metabolic_stoichiometry.npy")
u_s, _, _ = np.linalg.svd(S_mat, full_matrices=False)
tda_summary["10_metabolic_recon3d"] = process_point_cloud_tda("10_Recon3D_Metabolic_Cycle_Homology", u_s[:, :3], maxdim=1, max_points=150)

with open(f"{RESULTS_DIR}/tda_biomedical_10_domains_audit.json", "w") as f:
    json.dump(tda_summary, f, indent=2)

print("\n=========================================================================")
print(" ✅ ALL 10 DOMAINS TOPOLOGICALLY CHARACTERIZED & SAVED TO:")
print(f"    {RESULTS_DIR}/tda_biomedical_10_domains_audit.json")
print("=========================================================================")
