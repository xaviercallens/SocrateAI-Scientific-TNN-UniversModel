import os
import numpy as np
import pandas as pd
from pathlib import Path

DESI_LOCAL_DIR = Path("/mnt/disks/disk-socrateai-local-1/callensxavier_home_data/SocrateAI-Scientific-Agora-Home_data/raw/desi_dr1_noirlab")

def load_desi_catalog(catalog_type="lrg", max_samples=None):
    if not DESI_LOCAL_DIR.exists():
        raise FileNotFoundError(f"DESI local directory not found: {DESI_LOCAL_DIR}. Ensure disk is mounted.")
        
    csv_files = list(DESI_LOCAL_DIR.glob(f"{catalog_type.lower()}_zpix_photometry_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found for catalog type '{catalog_type}' in {DESI_LOCAL_DIR}")
        
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path, nrows=max_samples)
    df = df[(df['z'] > 0.0) & (df['z'] < 5.0)]
    return df

def desi_to_cartesian(df):
    ra_rad = np.radians(df['ra'].values)
    dec_rad = np.radians(df['dec'].values)
    z = df['z'].values
    R = z 
    x = R * np.cos(dec_rad) * np.cos(ra_rad)
    y = R * np.cos(dec_rad) * np.sin(ra_rad)
    z_c = R * np.sin(dec_rad)
    coords = np.column_stack([x, y, z_c])
    return coords

if __name__ == "__main__":
    print("Generating 3D Coordinates from DESI DR1...")
    df = load_desi_catalog("lrg", max_samples=500000)
    coords = desi_to_cartesian(df)
    np.save("scripts/real_data/desi_3d_coords.npy", coords)
    print(f"Saved {len(coords)} coordinates to scripts/real_data/desi_3d_coords.npy")
