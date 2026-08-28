import os
import json
import numpy as np
from pathlib import Path
from ripser import ripser

DATA_DIR = Path("/mnt/disks/disk-socrateai-local-1/dual_scale_datasets")
if not DATA_DIR.exists():
    DATA_DIR = Path("./data/real/dual_scale_datasets")

AUDIT_DIR = Path("tda_repository")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def compute_entropy(diagram):
    if len(diagram) == 0:
        return 0.0
    lifetimes = []
    for p in diagram:
        b, d = p[0], p[1]
        if np.isfinite(d) and d > b:
            lifetimes.append(d - b)
    if not lifetimes:
        return 0.0
    tot = sum(lifetimes)
    if tot == 0:
        return 0.0
    probs = [l / tot for l in lifetimes]
    return -sum(p * np.log(p) for p in probs if p > 0)

print("=========================================================================")
print(" COMPUTING TDA TOPOLOGICAL BARCODES FOR 10 DUAL-SCALE DOMAINS")
print("=========================================================================")

results = {}

# 1. Quantum Topological Materials
print("[1/10] TDA for Quantum Topological Materials...")
d1 = np.load(DATA_DIR / "domain01_topological_materials_berry.npz")
berry_z = d1["berry_z"]
coords1 = np.argwhere(np.abs(berry_z) > np.percentile(np.abs(berry_z), 90))[:200]
diag1 = ripser(coords1, maxdim=1)["dgms"]
results["domain_01_topological_materials"] = {
    "domain": "Quantum Topological Materials & Spintronics",
    "b0_degeneracy_clusters": int(len(diag1[0])),
    "b1_chern_loops": int(len(diag1[1])),
    "h1_entropy": float(compute_entropy(diag1[1])),
    "points_analyzed": len(coords1)
}

# 2. Plasma Turbulence & Tokamak Confinement
print("[2/10] TDA for Plasma Turbulence & Tokamak Gyrokinetics...")
d2 = np.load(DATA_DIR / "domain02_plasma_tokamak_gyrokinetics.npz")
pot = d2["potential"]
coords2 = np.argwhere(pot > np.percentile(pot, 85))[:200]
diag2 = ripser(coords2, maxdim=1)["dgms"]
results["domain_02_plasma_tokamak"] = {
    "domain": "Plasma Turbulence & Tokamak Fusion",
    "b0_flux_tubes": int(len(diag2[0])),
    "b1_drift_vortex_loops": int(len(diag2[1])),
    "h1_entropy": float(compute_entropy(diag2[1])),
    "points_analyzed": len(coords2)
}

# 3. High-Entropy Alloys & Dislocation Networks
print("[3/10] TDA for High-Entropy Alloys & Dislocation Networks...")
d3 = np.load(DATA_DIR / "domain03_hea_dislocations_cantor.npz")
nodes3 = d3["nodes"][:200]
diag3 = ripser(nodes3, maxdim=1)["dgms"]
results["domain_03_hea_dislocations"] = {
    "domain": "High-Entropy Alloys & Fracture Dislocation",
    "b0_dislocation_pileups": int(len(diag3[0])),
    "b1_frank_read_loops": int(len(diag3[1])),
    "h1_entropy": float(compute_entropy(diag3[1])),
    "points_analyzed": len(nodes3)
}

# 4. Multiscale Neurodynamics & Connectomics
print("[4/10] TDA for Multiscale Neurodynamics & Connectomics...")
d4 = np.load(DATA_DIR / "domain04_hcp_connectome_neurodynamics.npz")
coords4 = d4["coords"]
diag4 = ripser(coords4, maxdim=1)["dgms"]
results["domain_04_connectomics_neurodynamics"] = {
    "domain": "Multiscale Neurodynamics & Connectomics",
    "b0_functional_cliques": int(len(diag4[0])),
    "b1_synchronous_loops": int(len(diag4[1])),
    "h1_entropy": float(compute_entropy(diag4[1])),
    "points_analyzed": len(coords4)
}

# 5. Cloud Microphysics & Climate Teleconnections
print("[5/10] TDA for Cloud Microphysics & Planetary Climate...")
d5 = np.load(DATA_DIR / "domain05_cloudsat_radar_microphysics.npz")
rad5 = d5["reflectivity_dbz"]
coords5 = np.argwhere(rad5 > 15.0)[:200]
diag5 = ripser(coords5, maxdim=1)["dgms"]
results["domain_05_cloud_microphysics"] = {
    "domain": "Cloud Microphysics & Planetary Climate Teleconnections",
    "b0_convective_cores": int(len(diag5[0])),
    "b1_kelvin_helmholtz_rings": int(len(diag5[1])),
    "h1_entropy": float(compute_entropy(diag5[1])),
    "points_analyzed": len(coords5)
}

# 6. Superconducting Vortex Lattices & Fluxoid Pinning
print("[6/10] TDA for Superconducting Vortex Lattices...")
d6 = np.load(DATA_DIR / "domain06_supercon_abrikosov_vortices.npz")
coords6 = d6["coords"][:200]
diag6 = ripser(coords6, maxdim=1)["dgms"]
results["domain_06_superconductor_vortices"] = {
    "domain": "Superconducting Vortex Lattices & Fluxoid Pinning",
    "b0_abrikosov_vortex_cores": int(len(diag6[0])),
    "b1_flux_pinning_loops": int(len(diag6[1])),
    "h1_entropy": float(compute_entropy(diag6[1])),
    "points_analyzed": len(coords6)
}

# 7. Cardiovascular Hemodynamics & Thrombosis
print("[7/10] TDA for Cardiovascular Hemodynamics & Aorta WSS...")
d7 = np.load(DATA_DIR / "domain07_simvascular_aorta_wss.npz")
coords7 = d7["centerline"][:200]
diag7 = ripser(coords7, maxdim=1)["dgms"]
results["domain_07_cardiovascular_wss"] = {
    "domain": "Cardiovascular Hemodynamics & Plaque Thrombosis",
    "b0_arterial_landmarks": int(len(diag7[0])),
    "b1_secondary_vorticity_loops": int(len(diag7[1])),
    "h1_entropy": float(compute_entropy(diag7[1])),
    "points_analyzed": len(coords7)
}

# 8. Seismology, Fault Mechanics & Geodynamics
print("[8/10] TDA for Seismology & Fault Rupture...")
d8 = np.load(DATA_DIR / "domain08_earthscope_seismic_rupture.npz")
coords8 = d8["hypocenters"][:200]
diag8 = ripser(coords8, maxdim=1)["dgms"]
results["domain_08_seismology_rupture"] = {
    "domain": "Seismology, Fault Mechanics & Geodynamics",
    "b0_aftershock_hypocenters": int(len(diag8[0])),
    "b1_fault_system_loops": int(len(diag8[1])),
    "h1_entropy": float(compute_entropy(diag8[1])),
    "points_analyzed": len(coords8)
}

# 9. Multiphase Porous Media Flow & CO2 Sequestration
print("[9/10] TDA for Multiphase Porous Media Flow (Berea Sandstone)...")
d9 = np.load(DATA_DIR / "domain09_digitalrocks_berea_co2.npz")
mask9 = d9["porosity_mask"]
coords9 = np.argwhere(mask9 > 0.5)[:200]
diag9 = ripser(coords9, maxdim=1)["dgms"]
results["domain_09_porous_media_co2"] = {
    "domain": "Multiphase Granular & Porous Media Flow",
    "b0_isolated_pore_bodies": int(len(diag9[0])),
    "b1_percolation_throat_channels": int(len(diag9[1])),
    "h1_entropy": float(compute_entropy(diag9[1])),
    "points_analyzed": len(coords9)
}

# 10. Phononic Metamaterials & Acoustic Bandgaps
print("[10/10] TDA for Phononic Metamaterials & Acoustic Bandgaps...")
d10 = np.load(DATA_DIR / "domain10_mit_topopt_phononic_metamaterial.npz")
bands10 = d10["bands"]
k_p10 = d10["k_path"]
coords10 = np.column_stack([np.repeat(k_p10, 4), bands10.flatten()[:len(k_p10)*4]])[:200]
diag10 = ripser(coords10, maxdim=1)["dgms"]
results["domain_10_phononic_metamaterials"] = {
    "domain": "Phononic / Photonic Metamaterials",
    "b0_bloch_eigenstate_branches": int(len(diag10[0])),
    "b1_bandgap_spectral_loops": int(len(diag10[1])),
    "h1_entropy": float(compute_entropy(diag10[1])),
    "points_analyzed": len(coords10)
}

out_file = AUDIT_DIR / "dual_scale_10_domains_audit.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print("Saved TDA audit successfully to:", out_file)
