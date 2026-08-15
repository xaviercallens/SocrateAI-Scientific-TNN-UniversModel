import os
import json

def inventory_workspace(root_dir):
    inventory = {
        "lib_repositories": {},
        "specs_count": 0,
        "scripts_count": 0,
        "certs_count": 0,
        "total_files": 0
    }
    
    lib_path = os.path.join(root_dir, "lib")
    if os.path.exists(lib_path):
        for repo_name in os.listdir(lib_path):
            repo_full_path = os.path.join(lib_path, repo_name)
            if os.path.isdir(repo_full_path):
                file_count = 0
                py_count = 0
                for r, d, f in os.walk(repo_full_path):
                    if '.git' in d:
                        d.remove('.git')
                    file_count += len(f)
                    py_count += sum(1 for file in f if file.endswith('.py'))
                inventory["lib_repositories"][repo_name] = {
                    "total_files": file_count,
                    "python_files": py_count,
                    "path": repo_full_path
                }
                
    # Count specs, scripts, certs, datasets
    for folder in ["specs", "scripts", "certs"]:
        p = os.path.join(root_dir, folder)
        if os.path.exists(p):
            count = sum(len(files) for _, _, files in os.walk(p))
            inventory[f"{folder}_count"] = count

    dataset_path = os.path.join(root_dir, "data", "open_datasets")
    if os.path.exists(dataset_path):
        inventory["datasets_count"] = sum(len(files) for _, _, files in os.walk(dataset_path))
            
    return inventory

if __name__ == "__main__":
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inv = inventory_workspace(workspace_root)
    
    inventory_md = f"""# INVENTAIRE COMPLET DE LA BIBLIOTHÈQUE ET DU REPOSITOIRE

## 1. Dépôts Téléchargés dans `./lib/`
"""
    for repo, details in inv["lib_repositories"].items():
        inventory_md += f"- **{repo}** : {details['total_files']} fichiers au total ({details['python_files']} scripts Python)\n"
        
    inventory_md += f"""
## 2. Jeux de Données Ouverts (Datasets pour LAB-4)
- **Datasets (`/data/open_datasets`)** : {inv.get('datasets_count', 0)} fixtures de jeux de données (Shallow Water, SPID, JHTDB)

## 3. Infrastructure locale du projet
- **Spécifications (`/specs`)** : {inv.get('specs_count', 0)} fichiers d'architecture et mémos (LAB-0 à LAB-4, DSHL)
- **Pilotes & Simulateurs (`/scripts`)** : {inv.get('scripts_count', 0)} scripts Python (Drivers DSHL, TNN, Master Suite)
- **Certificats d'Audit (`/certs`)** : {inv.get('certs_count', 0)} certificats JSON d'audit cryptographique

*Généré automatiquement et certifié par l'Observatoire SocrateAI.*
"""
    
    inv_md_path = os.path.join(workspace_root, "lib", "INVENTORY.md")
    with open(inv_md_path, "w", encoding="utf-8") as f:
        f.write(inventory_md)
        
    print(inventory_md)
