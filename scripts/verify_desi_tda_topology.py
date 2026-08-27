"""
TDA OF THE COSMIC WEB: DESI DR1 REAL OBSERVATIONAL DATA
=======================================================
Phase 4: Astrophysical Gravitation & Topology

This script performs Topological Data Analysis (TDA) using Cubical Persistence
on the 3D Cosmic Web density field generated directly from the DESI DR1 LRG catalog.

It computes the Betti numbers (b0: Connected components, b1: Filaments/Loops, b2: Voids)
of the observed Dark Matter structure of the Universe, bridging topological invariants
with real astrophysics.
"""

import sys, os, time
import numpy as np
from gtda.homology import CubicalPersistence

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from real_data.desi_connector import fetch_desi_voxel_grid

def filter_persistent_features(diagram, threshold=0.1):
    features = []
    for point in diagram:
        birth, death, dim = point
        lifetime = death - birth
        if lifetime > threshold and dim != -1:  # Ignore padded noise
            features.append({'dim': int(dim), 'birth': float(birth), 'death': float(death), 'lifetime': float(lifetime)})
    return features

def extract_betti_numbers(features):
    betti = {0: 0, 1: 0, 2: 0}
    for f in features:
        if f['dim'] in betti:
            betti[f['dim']] += 1
    return betti

def verify_desi_tda():
    print("=======================================================================")
    print(" SOCRATE-AI TNN UNIVERS MODEL — DESI COSMIC WEB TOPOLOGY (TDA)")
    print("=======================================================================")
    start = time.time()
    
    # 1. Fetch DESI Voxel Grid
    print("\n[Phase 1] Loading DESI DR1 Cosmic Web Voxel Grid (Real Data)...")
    # Resolution 64x64x64 to get rich filamentary structure
    grid_size = 64
    grid = fetch_desi_voxel_grid(catalog_type="lrg", grid_size=grid_size, max_samples=500000)
    
    # In Cubical Persistence, sublevel sets look at values <= threshold.
    # Our grid has high values for high density. We want to find topology of the dense structures.
    # Therefore, we invert the grid so high density is low (appears early in filtration).
    inv_grid = 1.0 - grid
    
    print(f"  Voxel Grid Shape: {inv_grid.shape}")
    
    # 2. Compute Cubical Persistence
    print("\n[Phase 2] Computing Cubical Persistence (TDA)...")
    homologies = [0, 1, 2]
    # We reshape to (1, X, Y, Z) for giotto-tda
    cp = CubicalPersistence(homology_dimensions=homologies, n_jobs=-1)
    diagrams = cp.fit_transform(inv_grid.reshape(1, *inv_grid.shape))
    diagram = diagrams[0]
    
    # 3. Filter Significant Features (Persistent Topology)
    # Threshold filters out local sampling noise (short lifetimes)
    threshold = 0.005
    significant_features = filter_persistent_features(diagram, threshold=threshold)
    betti = extract_betti_numbers(significant_features)
    
    print("\n[Phase 3] Topo-Invariants (Betti Numbers) of the DESI Dark Matter Web:")
    print(f"  b_0 (Clusters / Halos) : {betti[0]}")
    print(f"  b_1 (Cosmic Filaments) : {betti[1]}")
    print(f"  b_2 (Cosmic Voids)     : {betti[2]}")
    
    # Physical Interpretation
    print("\n[Astrophysical Interpretation]")
    if betti[0] > 0 and betti[1] > 0 and betti[2] > 0:
        print("  ✅ The DESI data exhibits a true 'Cosmic Web' topology.")
        print(f"     It contains {betti[0]} primary dense clusters (Dark Matter Halos),")
        print(f"     connected by {betti[1]} significant filaments/bridges,")
        print(f"     enclosing {betti[2]} large-scale cosmic voids.")
    else:
        print("  ⚠️ The grid does not display full 3D cosmic web homology at this resolution/threshold.")
    
    dur = time.time() - start
    print("\n=======================================================================")
    print(f" ✅ DESI TOPOLOGY VERIFICATION COMPLETE | {dur:.2f}s")
    print("=======================================================================")

if __name__ == "__main__":
    verify_desi_tda()
