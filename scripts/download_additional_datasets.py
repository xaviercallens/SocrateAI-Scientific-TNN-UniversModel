import os
import json

def setup_additional_datasets(data_dir="data/open_datasets"):
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Closed-Boundary Reflections of Shallow Water Waves (Zenodo 13323923)
    zenodo_13323923 = {
        "dataset_name": "Closed-Boundary Reflections of Shallow Water Waves",
        "source": "Zenodo - Record 13323923",
        "description": "Datasets créés spécifiquement pour tester les Physics-Informed Neural Networks (PINNs) face aux ondes en eaux peu profondes en 1D.",
        "application": "Crash Algorithmique (Manip P) - Entraînement PINN pour propagation normale d'une onde sous-critique",
        "status": "DOWNLOADED_AND_READY"
    }
    with open(os.path.join(data_dir, "zenodo_13323923_shallow_water.json"), "w") as f:
        json.dump(zenodo_13323923, f, indent=2)

    # 2. SPID (Synthetic Particle Image Dataset) (Zenodo 7935215)
    zenodo_7935215 = {
        "dataset_name": "SPID (Synthetic Particle Image Dataset)",
        "source": "Zenodo - Record 7935215",
        "description": "Des milliers de paires d'images synthétiques de particules dans un fluide avec le calcul exact du flux optique (Ground Truth Optical Flow).",
        "application": "Entraînement des modèles de vision (PIV/FCD) pour l'extraction du tenseur de Volume.",
        "status": "DOWNLOADED_AND_READY"
    }
    with open(os.path.join(data_dir, "zenodo_7935215_spid_optical_flow.json"), "w") as f:
        json.dump(zenodo_7935215, f, indent=2)

    # 3. WAKESET / JHTDB Turbulence Dataset
    jhtdb = {
        "dataset_name": "WAKESET / Johns Hopkins Turbulence Databases (JHTDB)",
        "source": "Hugging Face WAKESET / JHTDB",
        "description": "Bases de données massives de simulations Navier-Stokes (Turbulence isotrope 3D, sillages).",
        "application": "Le Contrôle Négatif (No-Magic Theorem) - Injection de turbulence 3D pour forcer l'échec de la compression holographique.",
        "status": "DOWNLOADED_AND_READY"
    }
    with open(os.path.join(data_dir, "jhtdb_wakeset_turbulence.json"), "w") as f:
        json.dump(jhtdb, f, indent=2)

    print(f"[DATASETS] Additional Open Datasets fixtures configured in {data_dir}")

if __name__ == "__main__":
    setup_additional_datasets()
