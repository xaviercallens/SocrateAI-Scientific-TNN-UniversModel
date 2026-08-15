# MÉMO LAB-3 : Canal à Horizon Blanc (Banc CHOP)
*Status: Pre-registration protocols and hardware specifications*

---

## [Français] MÉMO LAB-3 : Canal à Horizon Blanc (Banc CHOP)

Plan d'implémentation du MEMO LAB-3, passant de l'idée géniale à un véritable observatoire cyber-physique automatisé : le Canal à Horizon Blanc (Banc CHOP).

### 1. Nomenclature et Liste de Matériel (BOM - Budget ~45 €)
Nous écartons le vortex de baignoire (spectaculaire mais instable et très complexe à instrumenter numériquement) au profit du canal 1D de Rousseaux (2008), beaucoup plus rigoureux.

| Catégorie | Composant | Rôle | Source / Coût |
| :--- | :--- | :--- | :--- |
| **Le Bulk (Espace)** | Gouttière PVC rectangulaire ou bac transparent (L: 1m, l: ~15cm) | Le canal d'espace-temps 1D. | ~10 € |
| **La Métrique** | Obstacle profilé (bosse douce type dos d'âne) en argile ou impression 3D | Réduit la section du canal pour accélérer le fluide ($v > c$). | 0 € |
| **Le Flux (Temps)** | Pompe d'aquarium submersible (1000 - 2000 L/h) | Crée le flot stationnaire (écoulement d'entraînement). | ~15 € |
| **Sonde (UV)** | Haut-parleur de récup' + petite tige en plastique collée au centre | Générateur d'ondes planes piloté par le Pi. | Récup (0 €) |
| **Traceur** | Lait ou colorant blanc (Dioxyde de Titane) | Rend l'eau opaque pour réflexion de surface. | Cuisine |
| **Optique** | Niveau laser "Ligne" de bricolage | Projette une ligne droite continue sur la surface de l'eau. | ~15 € |
| **Cerveau** | Raspberry Pi + Module Caméra | Jumeau Numérique et Système d'Observation. | Existant |

### 2. Architecture Matérielle et Schémas Électroniques
Le Raspberry Pi doit avoir l'autorité totale pour garantir l'horodatage de la prédiction avant l'activation.

#### A. Le Banc Hydraulique (Circulation)
La pompe est à une extrémité, un tuyau ramène l'eau à l'autre. Placez des pailles entassées (nid d'abeille) à la sortie amont pour "peigner" le flux : il doit être laminaire. La bosse est au milieu. Le haut-parleur est en aval et bat la surface de l'eau, envoyant des ondes à contre-courant vers la bosse.

#### B. Circuit de Commande (Hardware Layer)
- **Contrôle de la Sonde :** La sortie audio du Raspberry Pi (ou un GPIO en PWM) est connectée à un petit amplificateur audio de récupération (ex: module PAM8403 à 2€), lui-même relié au haut-parleur. Cela permet au Pi de générer une sinusoïde parfaite (ex: 3 Hz).
- **Observation Optique :** Le laser est fixé en hauteur, à un angle rasant (ex: 30°), projetant une ligne sur l'axe longitudinal du canal. La caméra filme strictement à la verticale (zénith). Une variation de hauteur de la vague ($\Delta h$) décalera visuellement la ligne laser latéralement, amplifiant le signal.

### 3. Le Jumeau Numérique (Simulateur et Prédiction)
Dans l'approximation des eaux peu profondes, la vitesse de l'onde est $c = \sqrt{gh}$ et la vitesse du fluide est $v = q/h$ (où $q$ est le débit linéique). L'horizon analogue se trouve exactement là où le nombre de Froude $Fr = v/c = 1$. L'onde est alors figée dans le référentiel du laboratoire.

**Script Python `predict_horizon.py` :**
(Le débit $Q$ se chronomètre avec un seau gradué avant l'expérience).

```python
import numpy as np
import scipy.optimize as opt
import json
import hashlib

# 1. Paramètres physiques
g = 9.81
largeur_b = 0.15          # Largeur du canal (15 cm)
Q_pompe = 0.00028         # Débit mesuré (~1000 L/h en m^3/s)
q = Q_pompe / largeur_b
E_tot = 0.05              # Énergie spécifique (liée à la profondeur au repos)

# 2. Modèle de la Métrique (L'obstacle)
def Z_obstacle(x):
    return 0.03 * np.exp(-((x - 0.5)**2) / 0.015) # Bosse centrée en x=0.5m

def bernoulli(h, x):
    # Conservation de l'énergie de Saint-Venant
    return h + Z_obstacle(x) + (q**2) / (2 * g * h**2) - E_tot

# 3. Calcul du Point de Bascule
x_array = np.linspace(0.3, 0.7, 500)
x_horizon_pred = None

for x in x_array:
    # Résolution de la hauteur d'eau locale h(x)
    h_sol = opt.fsolve(bernoulli, x0=0.04, args=(x,))[0]
    Fr = (q / h_sol) / np.sqrt(g * h_sol) # Froude = v / c
    if Fr >= 1.0 and x_horizon_pred is None:
        x_horizon_pred = x

# 4. Discipline LAB-0 : Verrouillage cryptographique
meta = {"horizon_pred_mm": round(x_horizon_pred*1000, 1), "status": "LOCKED"}
hash_val = hashlib.sha256(json.dumps(meta).encode()).hexdigest()
with open("lab3_run.meta", "w") as f:
    json.dump({"data": meta, "hash": hash_val}, f)
print(f"Prédiction validée : Horizon causal à {x_horizon_pred*1000:.1f} mm")
```

### 4. Système d'Observation Numérique : Le Kymographe
Le génie de ce protocole est de ne pas utiliser d'algorithme de tracking complexe ("boîte noire"), mais d'extraire un **Diagramme Espace-Temps de Minkowski (Kymographe)** pour prouver visuellement l'arrêt du temps local pour l'onde.

**Protocole (`generate_kymograph.py`) :**
1. La caméra tourne à 60 fps pendant 10 secondes.
2. Pour chaque image, le script extrait uniquement la ligne de pixels correspondant à l'axe longitudinal du canal (la ligne rouge du laser déformée par l'eau).
3. Le script empile ces tranches 1D verticalement. Vous obtenez une image 2D où l'axe X est l'espace, et l'axe Y est le temps.

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('run_P4.mp4')
kymograph = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # Extraction du canal Rouge sur la ligne centrale
    slice_1d = frame[int(frame.shape[0]/2), :, 2] 
    kymograph.append(slice_1d)

cv2.imwrite('kymograph_P4.png', np.vstack(kymograph))
```

**Comment lire la preuve P4 sur le Kymographe ?**
- Les vagues qui se propagent apparaissent comme des lignes obliques. La pente est la vitesse de l'onde ($c-v$).
- À l'approche de la bosse, $v$ augmente, la pente s'aplatit.
- **La Preuve :** À la coordonnée exacte prédite par le jumeau numérique, les lignes obliques percutent une asymptote strictement verticale. La ligne verticale signifie $dx/dt = 0$ : la vague est figée. L'information ne passe plus la frontière.

### 5. Protocole Expérimental et Contrôles Négatifs (Hard Constraints)
Votre labo doit prouver que le blocage est cinématique (lié au tenseur métrique du fluide) et non matériel (les vagues ne butent pas physiquement contre la bosse en argile).

- **Contrôle Négatif 1 (Espace Plat) :** Pompe éteinte ($v=0$). Haut-parleur ON. Le kymographe montre des droites obliques parfaites traversant tout le canal.
- **Contrôle Négatif 2 (Espace Sub-critique) :** Pompe allumée, mais débit réduit de moitié par une vanne ou une tension plus basse. Le Jumeau Numérique annonce $Fr < 1$ partout (Pas d'horizon). Le kymographe montrera des ondes qui ralentissent sur la bosse (pente courbée) mais qui finissent par passer de l'autre côté.
- **L'Épreuve P4 (Le Trou Blanc) :** Pompe nominale. L'onde se fracasse sur l'asymptote verticale. Mieux encore : sur le kymographe, juste avant l'horizon, vous verrez les lignes obliques se resserrer dramatiquement (Blueshift divergent) témoignant du rebond dispersif UV : l'onde transfère son énergie dans une échelle plus petite car la grande échelle lui est interdite.

---
---

## [English] MEMO LAB-3: White Hole Horizon Channel (CHOP Bench)

Implementation plan for MEMO LAB-3, transitioning from a brilliant idea to a true automated cyber-physical observatory: the White Hole Horizon Channel (CHOP Bench).

### 1. Nomenclature and Bill of Materials (BOM - Budget ~45 €)
We discard the bathtub vortex (spectacular but unstable and very complex to instrument digitally) in favor of Rousseaux's 1D channel (2008), which is much more rigorous.

| Category | Component | Role | Source / Cost |
| :--- | :--- | :--- | :--- |
| **The Bulk (Space)** | Rectangular PVC gutter or transparent tray (L: 1m, w: ~15cm) | The 1D spacetime channel. | ~10 € |
| **The Metric** | Profiled obstacle (smooth speed-bump type) in clay or 3D printed | Reduces the channel cross-section to accelerate the fluid ($v > c$). | 0 € |
| **The Flow (Time)** | Submersible aquarium pump (1000 - 2000 L/h) | Creates the stationary background flow. | ~15 € |
| **Probe (UV)** | Salvaged speaker + small plastic rod glued to the center | Plane wave generator driven by the Pi. | Salvaged (0 €) |
| **Tracer** | Milk or white dye (Titanium Dioxide) | Renders the water opaque for surface reflection. | Kitchen |
| **Optics** | DIY "Line" laser level | Projects a continuous straight line on the water surface. | ~15 € |
| **Brain** | Raspberry Pi + Camera Module | Digital Twin and Observation System. | Existing |

### 2. Hardware Architecture and Electronic Schematics
The Raspberry Pi must have total authority to guarantee the timestamping of the prediction before activation.

#### A. The Hydraulic Bench (Circulation)
The pump is at one end, a hose returns the water to the other. Place bundled straws (honeycomb) at the upstream outlet to "comb" the flow: it must be laminar. The bump is in the middle. The speaker is downstream and beats the water surface, sending counter-current waves towards the bump.

#### B. Control Circuit (Hardware Layer)
- **Probe Control:** The audio output of the Raspberry Pi (or a PWM GPIO) is connected to a small salvaged audio amplifier (e.g., PAM8403 module at 2€), itself connected to the speaker. This allows the Pi to generate a perfect sine wave (e.g., 3 Hz).
- **Optical Observation:** The laser is mounted high up, at a grazing angle (e.g., 30°), projecting a line along the longitudinal axis of the channel. The camera films strictly vertically (zenith). A variation in wave height ($\Delta h$) will visually shift the laser line laterally, amplifying the signal.

### 3. The Digital Twin (Simulator and Prediction)
In the shallow-water approximation, the wave speed is $c = \sqrt{gh}$ and the fluid velocity is $v = q/h$ (where $q$ is the linear discharge). The analogue horizon is located exactly where the Froude number $Fr = v/c = 1$. The wave is then frozen in the laboratory frame.

**Python Script `predict_horizon.py`:**
(The flow rate $Q$ is timed with a graduated bucket before the experiment).

```python
import numpy as np
import scipy.optimize as opt
import json
import hashlib

# 1. Physical parameters
g = 9.81
largeur_b = 0.15          # Channel width (15 cm)
Q_pompe = 0.00028         # Measured flow rate (~1000 L/h in m^3/s)
q = Q_pompe / largeur_b
E_tot = 0.05              # Specific energy (related to rest depth)

# 2. Metric Model (The obstacle)
def Z_obstacle(x):
    return 0.03 * np.exp(-((x - 0.5)**2) / 0.015) # Bump centered at x=0.5m

def bernoulli(h, x):
    # Saint-Venant energy conservation
    return h + Z_obstacle(x) + (q**2) / (2 * g * h**2) - E_tot

# 3. Calculation of the Tipping Point
x_array = np.linspace(0.3, 0.7, 500)
x_horizon_pred = None

for x in x_array:
    # Resolution of local water height h(x)
    h_sol = opt.fsolve(bernoulli, x0=0.04, args=(x,))[0]
    Fr = (q / h_sol) / np.sqrt(g * h_sol) # Froude = v / c
    if Fr >= 1.0 and x_horizon_pred is None:
        x_horizon_pred = x

# 4. LAB-0 Discipline: Cryptographic Locking
meta = {"horizon_pred_mm": round(x_horizon_pred*1000, 1), "status": "LOCKED"}
hash_val = hashlib.sha256(json.dumps(meta).encode()).hexdigest()
with open("lab3_run.meta", "w") as f:
    json.dump({"data": meta, "hash": hash_val}, f)
print(f"Prediction validated: Causal horizon at {x_horizon_pred*1000:.1f} mm")
```

### 4. Digital Observation System: The Kymograph
The genius of this protocol is not to use a complex tracking algorithm ("black box"), but to extract a **Minkowski Spacetime Diagram (Kymograph)** to visually prove the stopping of local time for the wave.

**Protocol (`generate_kymograph.py`):**
1. The camera runs at 60 fps for 10 seconds.
2. For each frame, the script extracts only the line of pixels corresponding to the longitudinal axis of the channel (the red laser line deformed by the water).
3. The script stacks these 1D slices vertically. You get a 2D image where the X-axis is space, and the Y-axis is time.

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('run_P4.mp4')
kymograph = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # Extraction of the Red channel on the center line
    slice_1d = frame[int(frame.shape[0]/2), :, 2] 
    kymograph.append(slice_1d)

cv2.imwrite('kymograph_P4.png', np.vstack(kymograph))
```

**How to read the P4 proof on the Kymograph?**
- Propagating waves appear as oblique lines. The slope is the wave speed ($c-v$).
- Approaching the bump, $v$ increases, the slope flattens.
- **The Proof:** At the exact coordinate predicted by the digital twin, the oblique lines hit a strictly vertical asymptote. The vertical line means $dx/dt = 0$: the wave is frozen. Information no longer crosses the boundary.

### 5. Experimental Protocol and Negative Controls (Hard Constraints)
Your lab must prove that the blockage is kinematic (tied to the fluid metric tensor) and not material (the waves do not physically crash into the clay bump).

- **Negative Control 1 (Flat Space):** Pump off ($v=0$). Speaker ON. The kymograph shows perfect oblique lines crossing the entire channel.
- **Negative Control 2 (Sub-critical Space):** Pump on, but flow halved by a valve or lower voltage. The Digital Twin announces $Fr < 1$ everywhere (No horizon). The kymograph will show waves slowing down over the bump (curved slope) but eventually passing to the other side.
- **The P4 Test (The White Hole):** Nominal pump. The wave crashes onto the vertical asymptote. Better still: on the kymograph, just before the horizon, you will see the oblique lines tighten dramatically (Divergent Blueshift), testifying to the UV dispersive bounce: the wave transfers its energy into a smaller scale because the large scale is forbidden to it.
