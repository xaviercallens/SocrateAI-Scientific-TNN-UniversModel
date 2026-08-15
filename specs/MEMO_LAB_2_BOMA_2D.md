# MÉMO LAB-2 : Banc Opto-Mécanique Automatisé 2D (BOMA-2D)
*Status: Pre-registration protocols and hardware specifications*

---

## [Français] MÉMO LAB-2 : Banc Opto-Mécanique Automatisé 2D (BOMA-2D)

Nous allons concevoir le Banc Opto-Mécanique Automatisé 2D (BOMA-2D), utilisant des chariots de lecteurs DVD croisés pour scanner le plan de Fourier (l'espace dual) au micromètre près.

### 1. Nomenclature et Liste de Matériel (BOM)
L'ingénierie d'un tel système exige de séparer la couche commande (mouvement) de la couche mesure (acquisition).

| Catégorie | Composant | Rôle | Source / Coût |
| :--- | :--- | :--- | :--- |
| **Mécanique** | 2 × Blocs optiques complets (CD/DVD) | Platines de translation X et Y. Contiennent les moteurs pas-à-pas et les vis sans fin. | Récupération (0 €) |
| **Calcul** | 1 × Raspberry Pi (ou Milk-V / RISC-V) | Cerveau de l'observatoire, hébergement du Jumeau Numérique. | Existant |
| **Commande** | 2 × Drivers A4988 ou DRV8825 | Pilotage micro-pas des moteurs DVD. | ~4 € |
| **Mesure** | 1 × CAN (ADC) ADS1115 (16-bit) | Numérisation haute précision de l'intensité lumineuse. | ~5 € |
| **Capteur** | 1 × Photodiode (ex: BPW34) | Mesure de l'intensité laser locale (réponse linéaire, contrairement aux caméras). | ~2 € |
| **Optique** | 2 × Lentilles convergentes (f ≈ 10-15 cm) | Construction du corrélateur 4f. | ~3 € |
| **Laser** | 1 × Module Diode Laser (5mW max) | Source cohérente (utiliser un module du commerce pour la sécurité, pas la diode nue du DVD). | ~5 € |

### 2. Schémas Électroniques et Câblage (Hardware Layer)
Nous évitons les caméras automatiques (trop de post-traitement non linéaire). L'observation se fera "au point" : la photodiode est déplacée physiquement en X et Y dans le plan d'observation.

#### A. Circuit de Commande (Moteurs X-Y)
Le Raspberry Pi pilote les drivers A4988 via ses broches GPIO (General Purpose Input/Output).
- **Alimentation Moteurs (VMOT) :** Bloc d'alimentation externe 5V-9V (ne jamais tirer la puissance des moteurs depuis le Pi). GND commun avec le Pi.
- **Logique (VDD) :** 3.3V (fourni par le Pi).

| Broche Driver A4988 (X) | Broche Driver A4988 (Y) | Raspberry Pi (GPIO BCM) |
| :--- | :--- | :--- |
| DIR (Direction) | - | GPIO 20 |
| STEP (Pas) | - | GPIO 21 |
| - | DIR (Direction) | GPIO 23 |
| - | STEP (Pas) | GPIO 24 |
| GND (Logique) | GND (Logique) | GND Physique (ex: Pin 6) |

#### B. Circuit de Mesure Quantique/Optique
Pour lire l'intensité lumineuse avec précision, la photodiode doit être montée en inverse (mode photoconducteur) avec une résistance, lue par l'ADC I2C. Pour réduire le bruit, l'ADC est placé au plus près de la diode.

| Composant | Connexion 1 | Connexion 2 | Raspberry Pi |
| :--- | :--- | :--- | :--- |
| Photodiode (BPW34) | Cathode (+3.3V Pi) | Anode (vers A0 sur ADS1115) | - |
| Résistance (10 kΩ) | Anode Photodiode | GND | - |
| ADS1115 (I2C) | VDD (+3.3V Pi) / GND | SDA / SCL | GPIO 2 (SDA) / GPIO 3 (SCL) |

*Astuce de conception :* La résistance de 10 kΩ convertit le très faible courant de la photodiode en tension lue par l'ADC ($V = R \cdot I$). Si le signal est trop faible, augmentez la résistance (ou ajoutez un amplificateur opérationnel en montage transimpédance).

### 3. Architecture Mécanique et Alignement
L'objectif est de recréer l'expérience Manip G (Corrélateur 4f) du Lab-1, mais de manière automatisée.

- **Le Banc Principal :** Un profilé d'aluminium (ou une planche de bois épaisse peinte en noir mat).
- **La Table XY (Le Capteur) :**
  1. Prenez le premier chariot de DVD (Axe X). Fixez-le rigidement sur la base.
  2. Prenez le second chariot (Axe Y). Fixez sa base sur la partie mobile du premier chariot, perpendiculairement.
  3. Fixez la photodiode (et un minuscule trou d'épingle/pinhole de papier aluminium devant elle pour augmenter la résolution spatiale) sur la partie mobile du chariot Y.
- **L'Axe Optique (Z) :**
  - $Z = 0$ : Laser + Fente ou Grille (l'objet matériel).
  - $Z = f$ : Lentille 1.
  - $Z = 2f$ : Plan de Fourier (C'est ici que la Table XY déplace la photodiode pour scanner l'espace dual).

### 4. Jumeau Numérique (Simulation Pré-Expérimentale)
Avant de lancer les moteurs, le système doit simuler la physique. L'environnement de recherche autonome doit forcer le calcul de la prédiction exacte.

**Protocole logiciel (Architecture Python) :**
1. Modéliser la géométrie de la fente ou de la grille (matrice 2D).
2. Appliquer la transformée de Fourier optique (l'approximation de Fraunhofer dicte que l'amplitude au plan focal est proportionnelle à la Transformée de Fourier spatiale de l'objet).
3. Calculer l'intensité ($I = |A|^2$).

**Squelette du script de simulation (`simulate_dual_space.py`) :**
```python
import numpy as np
import h5py

def simulate_fourier_plane(aperture_matrix, wavelength, focal_length, pixel_size):
    # Transformée de Fourier spatiale (FFT 2D)
    field_fourier = np.fft.fftshift(np.fft.fft2(aperture_matrix))
    
    # L'intensité est le module au carré de l'amplitude complexe
    intensity = np.abs(field_fourier)**2
    
    # Normalisation pour correspondre à la plage de l'ADC (0-65535 pour 16 bits)
    intensity_norm = (intensity / np.max(intensity)) * 65535
    return intensity_norm

# Paramètres physiques
grid_size = 500  # Résolution de la simulation
aperture = np.zeros((grid_size, grid_size))
aperture[240:260, 240:260] = 1.0  # Fente carrée

# Exécution de la simulation
predicted_intensity = simulate_fourier_plane(aperture, 650e-9, 0.1, 10e-6)

# Verrouillage : Écriture de la prédiction dans un fichier HDF5 immuable
with h5py.File("prediction_run_001.h5", "w") as f:
    f.create_dataset("expected_intensity", data=predicted_intensity)
    f.attrs["status"] = "PRE-REGISTERED"
```

### 5. Protocole d'Exécution et Système d'Observation
L'Observatoire Numérique orchestre la collision entre le Jumeau Numérique et la réalité physique via un script maître (`run_experiment.sh`).

#### A. La Règle d'Or (Hard Constraint)
Le script d'acquisition physique ne peut pas se lancer si le script de hachage ne trouve pas un fichier de simulation `prediction_run_XXX.h5` horodaté d'au moins 1 seconde avant l'heure actuelle.

#### B. Le Déroulé de l'Acquisition (Scan Spatial)
1. **Initialisation :** Le script de commande amène les moteurs XY en position (0,0) (butée mécanique ou capteur fin de course de récupération).
2. **Balayage (Raster Scan) :**
   - Pour $X$ de $0$ à $X_{max}$ par pas de 10 $\mu$m (un pas de moteur DVD équivaut souvent à ~3 à 15 $\mu$m).
   - Pour $Y$ de $0$ à $Y_{max}$ par pas de 10 $\mu$m.
   - Arrêt de 5 millisecondes.
   - Lecture I2C de l'ADS1115 (3 échantillons moyennés).
   - Enregistrement dans une structure de données en mémoire.
3. **Clôture :** Sécurisation du laser (extinction), retour au point zéro.

#### C. L'Analyse Post-Expérimentale (Le Résidu)
- Le système compare la matrice `mesure_physique` avec `prediction_run_XXX.h5`.
- Le script aligne les deux matrices (pour corriger les erreurs de centrage).
- Il calcule l'erreur quadratique moyenne (MSE) et trace la **carte des résidus** : une image 2D montrant exactement où la théorie a divergé de la réalité (généralement à cause des aberrations sphériques de vos lentilles à 3 €, ce qui est en soi une excellente donnée sur la courbure de l'espace optique local).

Ce Lab 2 vous donne une plateforme générique : en changeant simplement l'objet au plan $Z=0$ et le code du Jumeau Numérique, vous pouvez tester des réseaux de diffraction, des figures d'interférence complexes, ou des analogies de trous noirs optiques, avec une traçabilité digne d'un laboratoire institutionnel, pour moins de 25 euros.

---
---

## [English] MEMO LAB-2: Automated 2D Opto-Mechanical Bench (BOMA-2D)

We will design the Automated 2D Opto-Mechanical Bench (BOMA-2D), using crossed DVD drive sleds to scan the Fourier plane (the dual space) down to the micrometer.

### 1. Nomenclature and Bill of Materials (BOM)
Engineering such a system requires separating the control layer (movement) from the measurement layer (acquisition).

| Category | Component | Role | Source / Cost |
| :--- | :--- | :--- | :--- |
| **Mechanical** | 2 × Complete optical blocks (CD/DVD) | X and Y translation stages. Contain stepper motors and worm gears. | Salvaged (0 €) |
| **Compute** | 1 × Raspberry Pi (or Milk-V / RISC-V) | Brain of the observatory, hosting the Digital Twin. | Existing |
| **Control** | 2 × A4988 or DRV8825 drivers | Micro-step driving of the DVD motors. | ~4 € |
| **Measurement** | 1 × ADS1115 ADC (16-bit) | High-precision digitization of light intensity. | ~5 € |
| **Sensor** | 1 × Photodiode (e.g., BPW34) | Local laser intensity measurement (linear response, unlike cameras). | ~2 € |
| **Optics** | 2 × Converging lenses (f ≈ 10-15 cm) | Construction of the 4f correlator. | ~3 € |
| **Laser** | 1 × Laser Diode Module (5mW max) | Coherent source (use a commercial module for safety, not the bare DVD diode). | ~5 € |

### 2. Electronic Schematics and Wiring (Hardware Layer)
We avoid automatic cameras (too much nonlinear post-processing). Observation will be "at the point": the photodiode is physically moved in X and Y within the observation plane.

#### A. Control Circuit (X-Y Motors)
The Raspberry Pi drives the A4988 drivers via its GPIO (General Purpose Input/Output) pins.
- **Motor Power (VMOT):** External 5V-9V power supply (never draw motor power from the Pi). Shared GND with the Pi.
- **Logic (VDD):** 3.3V (provided by the Pi).

| Driver A4988 Pin (X) | Driver A4988 Pin (Y) | Raspberry Pi (GPIO BCM) |
| :--- | :--- | :--- |
| DIR (Direction) | - | GPIO 20 |
| STEP (Step) | - | GPIO 21 |
| - | DIR (Direction) | GPIO 23 |
| - | STEP (Step) | GPIO 24 |
| GND (Logic) | GND (Logic) | Physical GND (e.g., Pin 6) |

#### B. Quantum/Optical Measurement Circuit
To read light intensity accurately, the photodiode must be reverse-biased (photoconductive mode) with a resistor, read by the I2C ADC. To reduce noise, the ADC is placed as close to the diode as possible.

| Component | Connection 1 | Connection 2 | Raspberry Pi |
| :--- | :--- | :--- | :--- |
| Photodiode (BPW34) | Cathode (+3.3V Pi) | Anode (to A0 on ADS1115) | - |
| Resistor (10 kΩ) | Photodiode Anode | GND | - |
| ADS1115 (I2C) | VDD (+3.3V Pi) / GND | SDA / SCL | GPIO 2 (SDA) / GPIO 3 (SCL) |

*Design tip:* The 10 kΩ resistor converts the very small photodiode current into a voltage read by the ADC ($V = R \cdot I$). If the signal is too weak, increase the resistor (or add an operational amplifier in a transimpedance configuration).

### 3. Mechanical Architecture and Alignment
The goal is to recreate the Manip G (4f Correlator) experiment from Lab-1, but in an automated way.

- **Main Bench:** An aluminum extrusion (or a thick wooden board painted matte black).
- **The XY Table (The Sensor):**
  1. Take the first DVD sled (X Axis). Fix it rigidly to the base.
  2. Take the second sled (Y Axis). Fix its base onto the moving part of the first sled, perpendicularly.
  3. Attach the photodiode (and a tiny aluminum foil pinhole in front of it to increase spatial resolution) to the moving part of the Y sled.
- **The Optical Axis (Z):**
  - $Z = 0$: Laser + Slit or Grid (the material object).
  - $Z = f$: Lens 1.
  - $Z = 2f$: Fourier Plane (This is where the XY Table moves the photodiode to scan the dual space).

### 4. Digital Twin (Pre-Experimental Simulation)
Before running the motors, the system must simulate the physics. The autonomous research environment must force the calculation of the exact prediction.

**Software Protocol (Python Architecture):**
1. Model the geometry of the slit or grid (2D matrix).
2. Apply the optical Fourier transform (the Fraunhofer approximation dictates that the amplitude at the focal plane is proportional to the spatial Fourier Transform of the object).
3. Calculate the intensity ($I = |A|^2$).

**Simulation Script Skeleton (`simulate_dual_space.py`):**
```python
import numpy as np
import h5py

def simulate_fourier_plane(aperture_matrix, wavelength, focal_length, pixel_size):
    # Spatial Fourier Transform (2D FFT)
    field_fourier = np.fft.fftshift(np.fft.fft2(aperture_matrix))
    
    # Intensity is the squared magnitude of the complex amplitude
    intensity = np.abs(field_fourier)**2
    
    # Normalization to match the ADC range (0-65535 for 16-bit)
    intensity_norm = (intensity / np.max(intensity)) * 65535
    return intensity_norm

# Physical parameters
grid_size = 500  # Simulation resolution
aperture = np.zeros((grid_size, grid_size))
aperture[240:260, 240:260] = 1.0  # Square slit

# Simulation execution
predicted_intensity = simulate_fourier_plane(aperture, 650e-9, 0.1, 10e-6)

# Locking: Writing the prediction to an immutable HDF5 file
with h5py.File("prediction_run_001.h5", "w") as f:
    f.create_dataset("expected_intensity", data=predicted_intensity)
    f.attrs["status"] = "PRE-REGISTERED"
```

### 5. Execution Protocol and Observation System
The Digital Observatory orchestrates the collision between the Digital Twin and physical reality via a master script (`run_experiment.sh`).

#### A. The Golden Rule (Hard Constraint)
The physical acquisition script cannot launch if the hashing script does not find a simulation file `prediction_run_XXX.h5` timestamped at least 1 second before the current time.

#### B. The Acquisition Workflow (Spatial Scan)
1. **Initialization:** The control script brings the XY motors to position (0,0) (mechanical stop or salvaged limit switch).
2. **Raster Scan:**
   - For $X$ from $0$ to $X_{max}$ in steps of 10 $\mu$m (one DVD motor step is often ~3 to 15 $\mu$m).
   - For $Y$ from $0$ to $Y_{max}$ in steps of 10 $\mu$m.
   - Pause for 5 milliseconds.
   - I2C read from the ADS1115 (3 averaged samples).
   - Record into an in-memory data structure.
3. **Closure:** Secure the laser (turn off), return to the zero point.

#### C. Post-Experimental Analysis (The Residual)
- The system compares the `mesure_physique` matrix with `prediction_run_XXX.h5`.
- The script aligns the two matrices (to correct centering errors).
- It calculates the Mean Squared Error (MSE) and plots the **residual map**: a 2D image showing exactly where the theory diverged from reality (usually due to spherical aberrations in your 3 € lenses, which is in itself excellent data on the curvature of local optical space).

This Lab 2 gives you a generic platform: by simply changing the object at the $Z=0$ plane and the Digital Twin code, you can test diffraction gratings, complex interference patterns, or optical black hole analogies, with a traceability worthy of an institutional laboratory, for less than 25 euros.
