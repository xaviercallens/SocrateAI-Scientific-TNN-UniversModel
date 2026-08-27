"""
REAL DATA CONNECTOR: DESI (Dark Energy Spectroscopic Instrument) DR1
====================================================================
This connector loads the local DESI Early Data Release (DR1) galaxy catalogs
available on the local disk. These catalogs contain 3D spatial phase-space 
data (RA, Dec, Redshift) for various targets (LRG, ELG, QSO, BGS).

This data allows us to construct the 3D Cosmic Web and verify the 
topological features (Betti numbers) of the underlying dark matter 
distribution.

Complies with LL.md Étape 6 (Zero Synthetic Data Policy).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Path to the local DESI DR1 dataset on the secondary disk
DESI_LOCAL_DIR = Path("/mnt/disks/disk-socrateai-local-1/callensxavier_home_data/SocrateAI-Scientific-Agora-Home_data/raw/desi_dr1_noirlab")

def load_desi_catalog(catalog_type="lrg", max_samples=None):
    """
    Load a DESI DR1 galaxy catalog (LRG, ELG, QSO, or BGS).
    
    Parameters:
      catalog_type (str): The target type ('lrg', 'elg', 'qso', 'bgs').
      max_samples (int): Maximum number of rows to load (for memory/testing).
      
    Returns:
      pd.DataFrame: A DataFrame containing at least 'ra', 'dec', and 'z' (redshift).
    """
    if not DESI_LOCAL_DIR.exists():
        raise FileNotFoundError(f"DESI local directory not found: {DESI_LOCAL_DIR}. Ensure disk is mounted.")
        
    # Search for the CSV file matching the catalog type
    csv_files = list(DESI_LOCAL_DIR.glob(f"{catalog_type.lower()}_zpix_photometry_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found for catalog type '{catalog_type}' in {DESI_LOCAL_DIR}")
        
    csv_path = csv_files[0]
    print(f"[DESI] Loading {catalog_type.upper()} catalog from: {csv_path.name}")
    
    # Load data
    df = pd.read_csv(csv_path, nrows=max_samples)
    
    # Filter valid redshifts (z > 0 and zwarn == 0 usually indicates good spectra, though here we just want basic valid z)
    df = df[(df['z'] > 0.0) & (df['z'] < 5.0)]
    
    print(f"[DESI] Loaded {len(df)} valid {catalog_type.upper()} galaxies.")
    return df

def desi_to_cartesian(df):
    """
    Convert spherical coordinates (RA, Dec, Redshift z) to 3D Cartesian coordinates (X, Y, Z).
    Assumes a flat cosmology where comoving distance D_c is roughly proportional to z for low z.
    For more exact cosmological distances, integrate the FLRW metric.
    
    Returns:
       np.ndarray: [N, 3] array of Cartesian coordinates.
    """
    # Convert RA, Dec from degrees to radians
    ra_rad = np.radians(df['ra'].values)
    dec_rad = np.radians(df['dec'].values)
    z = df['z'].values
    
    # Approximate radial comoving distance (proportional to redshift for local universe)
    # R ~ z (in arbitrary units for topological clustering purposes)
    R = z 
    
    x = R * np.cos(dec_rad) * np.cos(ra_rad)
    y = R * np.cos(dec_rad) * np.sin(ra_rad)
    z_c = R * np.sin(dec_rad)
    
    coords = np.column_stack([x, y, z_c])
    return coords

def fetch_desi_voxel_grid(catalog_type="lrg", grid_size=32, max_samples=100000):
    """
    HIGH-LEVEL: Returns a physically grounded 3D cosmic web density field 
    from real DESI observational data.
    
    This converts the point cloud of galaxies into a voxelized 3D density grid,
    which can be fed directly into our Topo-Encoders to compute Betti numbers.
    """
    print(f"[DESI] Generating 3D Cosmic Web Voxel Grid (Size: {grid_size}^3)...")
    df = load_desi_catalog(catalog_type, max_samples=max_samples)
    coords = desi_to_cartesian(df)
    
    # Normalize coordinates to [0, 1] range for voxelization
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)
    
    # 3D Histogram to create density field
    edges = np.linspace(0, 1, grid_size + 1)
    density, _ = np.histogramdd(coords_norm, bins=[edges, edges, edges])
    
    # Smooth the point-mass density to create a continuous field (Dark Matter proxy)
    grid = gaussian_filter(density.astype(np.float64), sigma=1.0)
    
    # Normalize grid density to [0, 1]
    grid = grid / (grid.max() + 1e-8)
    
    print(f"[DESI] Generated density grid. Max density voxel: {grid.max():.4f}")
    return grid

if __name__ == "__main__":
    # Quick test
    grid = fetch_desi_voxel_grid(catalog_type="lrg", grid_size=32, max_samples=50000)
    print(f"Grid shape: {grid.shape}")
    print(f"Non-zero voxels (>0.1): {(grid > 0.1).sum()} / {grid.size}")
