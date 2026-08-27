"""
REAL DATA CONNECTOR: DESI (GCP Datalake)
========================================
This connector interfaces with the SocrateAI GCP Datalake to fetch 
the DESI Baryon Acoustic Oscillations (BAO) consensus covariance matrices
and observables.
"""
import os
import subprocess
import pandas as pd
import numpy as np

def fetch_gcp_desi_bao(bucket_path="gs://socrateai-datalake-gen-lang-client-0625573011/stream3_desi_dr1/"):
    print(f"[GCP DESI] Fetching BAO mean observables from {bucket_path}")
    
    # We download one of the mean BAO observable files as a test
    target_file = "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt"
    full_path = bucket_path + target_file
    local_path = f"/tmp/{target_file}"
    
    try:
        subprocess.run(["gcloud", "storage", "cp", full_path, local_path], check=True, capture_output=True)
        print(f"[GCP DESI] Successfully downloaded to {local_path}")
        
        # Load and parse the BAO mean values
        with open(local_path, 'r') as f:
            lines = f.readlines()
            
        data = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append({
                        'z': float(parts[0]),
                        'observable': float(parts[1])
                    })
        
        df = pd.DataFrame(data)
        print(f"[GCP DESI] Parsed {len(df)} BAO data points.")
        return df
        
    except subprocess.CalledProcessError as e:
        print(f"[GCP DESI Error] Failed to download {target_file}")
        print(e.stderr.decode('utf-8'))
        return None
        
if __name__ == "__main__":
    df = fetch_gcp_desi_bao()
    if df is not None:
        print(df.head())
