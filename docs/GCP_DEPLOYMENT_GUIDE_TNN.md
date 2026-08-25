# GUIDE DE DÉPLOIEMENT GCP (GPU T4) - ENTRAÎNEMENT TNN LABS 5-7

Ce document décrit la procédure complète pour migrer l'environnement de travail de votre machine locale (sans GPU) vers votre instance Google Cloud Platform (GCP) équipée d'un GPU NVIDIA T4.

## 1. Préparation de l'Instance GCP

Connectez-vous à votre instance GCP existante (celle utilisée précédemment pour la sélection K3) via SSH.

```bash
# Exemple de connexion SSH vers votre instance GCP
gcloud compute ssh nom-de-votre-instance --zone=votre-zone
```

## 2. Récupération du Code à Jour

Puisque nous venons de commiter et pusher toutes les architectures TNN des Labs 5, 6, et 7 ainsi que le tableau de bord maître, il vous suffit de récupérer les dernières modifications sur l'instance GCP.

Placez-vous dans le répertoire du projet sur la machine distante et lancez :

```bash
cd /chemin/vers/votre/repo/SocrateAI-Scientific-TNN-UniversModel
git pull origin main
```

*Si le dépôt n'est pas encore cloné sur l'instance :*
```bash
git clone https://github.com/xaviercallens/SocrateAI-Scientific-TNN-UniversModel.git
cd SocrateAI-Scientific-TNN-UniversModel
```

## 3. Configuration de l'Environnement Virtuel & Dépendances

Assurez-vous que l'environnement virtuel Python est activé et que PyTorch (avec support CUDA 11.8 / 12.1) est correctement installé pour profiter du GPU T4.

```bash
# Activer l'environnement virtuel (ou en créer un nouveau)
source .venv/bin/activate  # ou: python3 -m venv .venv && source .venv/bin/activate

# Vérifier la présence du GPU T4
nvidia-smi

# Installer les dépendances avec support CUDA
pip install -r requirements.txt
# Si PyTorch n'est pas installé avec CUDA, forcez l'installation :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 4. Vérification de l'Accès au GPU

Avant de lancer de gros entraînements, exécutez ce test rapide pour confirmer que PyTorch utilise bien le GPU T4 :

```bash
python3 -c "import torch; print(f'CUDA Disponible: {torch.cuda.is_available()}'); print(f'Carte: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Aucune\"}')"
```
Vous devriez voir : `Carte: Tesla T4`.

## 5. Exécution Initiale et Scripts d'Entraînement

Tous les modèles (Labs 0 à 7) sont conçus pour se router automatiquement vers `"cuda"` s'il est disponible.

**Étape A : Générer le registre initial**
```bash
python3 scripts/tnn_model_registry.py
```
*(Cela va créer les poids de départ pour tous les modèles dans le dossier `models/` en utilisant la puissance du GPU).*

**Étape B : Lancer des entraînements longs (Ex: LAB-5 TDA)**
Pour éviter que l'entraînement ne s'arrête si votre connexion SSH coupe, utilisez `tmux` ou `nohup`.

```bash
# Lancer un entraînement long en arrière-plan
nohup python3 scripts/lab5_long_run.py > logs/lab5_training_gcp.log 2>&1 &
```

Vous pouvez suivre la progression en direct avec :
```bash
tail -f logs/lab5_training_gcp.log
```

## 6. Lancement du Dashboard (Optionnel)

Si vous souhaitez exposer le Master Hub UI (Dashboard) depuis l'instance GCP pour le visualiser dans votre navigateur, lancez le serveur :

```bash
python3 scripts/start_dashboard_server.py --host 0.0.0.0 --port 8080
```
*Assurez-vous d'avoir ouvert le port 8080 dans les règles de pare-feu (Firewall rules) de votre projet Google Cloud.*

---
**STATUT DU SYSTÈME** : Prêt pour le transfert vers le Cloud GPU. La rigueur épistémique et le formalisme Lean 4 restent garantis.
