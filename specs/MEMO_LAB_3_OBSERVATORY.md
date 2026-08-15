# MÉMO LAB-3 : Observatoire Cyber-Physique de Gravité Analogue
*Status: Pre-registration protocols and hardware specifications*

---

## [Français] MÉMO LAB-3 : Observatoire Cyber-Physique de Gravité Analogue

Implémentation complète d'un Observatoire Cyber-Physique de Gravité Analogue, passant de l'idée géniale à un véritable instrument automatisé.

### 1. Le Saut Technologique : La Démodulation de Damier (FCD)
Nous abandonnons le laser rasant du LAB-2. Les articles mentionnent une technique redoutable, non-invasive et gratuite : la **FCD (Fast-Chequerboard Demodulation)** ou Synthetic Schlieren.

- **Principe :** Un motif en damier (carrés de 2 mm) est placé sous le fond transparent de la cuve, rétro-éclairé uniformément. La caméra du Raspberry Pi filme rigoureusement au zénith.
- **La Physique :** La surface de l'eau agit comme un champ de microlentilles. Une variation de hauteur $\nabla h$ dévie la lumière et déforme le damier apparent. Un algorithme de Flux Optique (Optical Flow) compare l'image déformée à l'image au repos et reconstruit la carte d'élévation 3D $\delta h(t, x, y)$ à 60 fps.

### 2. Nomenclature et Matériel (BOM - Budget ~50 €)
L'architecture sépare la création de la métrique (Pompe) de la perturbation quantique (Percuteur) et de la mesure (FCD).

| Catégorie | Composant | Rôle | Coût |
| :--- | :--- | :--- | :--- |
| **Bulk 2D** | Bac plat en acrylique (ex: 60x60 cm) percé au centre. | Héberge le Vortex d'Unruh. | ~15 € |
| **Bulk 1D** | Gouttière PVC avec inserts latéraux profilés (voir Manip N). | Héberge l'Horizon Conforme. | ~10 € |
| **Générateur IR** | Pompe d'aquarium (1500 L/h) + Vannes. | Fixe la métrique ($C$, $D$, Débit $Q$). | ~15 € |
| **Générateur UV** | Solénoïde 5V + tige en plastique, piloté par MOSFET. | Le percuteur (Le "Pluck" du trou noir). | ~5 € |
| **Optique FCD** | Damier imprimé + Dalle LED (récup d'un vieil écran LCD). | Mire de référence optique. | 0 € |
| **Cerveau** | Raspberry Pi 4 + Module Caméra (HQ ou V2). | Jumeau Numérique et Interféromètre. | Existant |

*(Câblage du percuteur : Le Pi pilote la grille d'un transistor MOSFET IRLZ44N, qui laisse passer la puissance vers le solénoïde pour garantir un impact temporellement parfait de 50 ms).*

### 3. MANIP M — Spectroscopie du Trou Noir (ABHS 2D)
**Objectif :** Valider l'holographie d'état. Retrouver la géométrie de l'écoulement (Circulation $C$, Drain $D$) uniquement en "écoutant" les ondes de surface.

#### Le Jumeau Numérique (`predict_abhs.py`) :
Le Pi calcule le spectre théorique des Light-Rings d'eau peu profonde (Équation 9 du premier document arXiv).

```python
import numpy as np

def theoretical_light_ring_freq(m, C, D, h_water):
    c_speed = np.sqrt(9.81 * h_water)
    sign = 1 if m > 0 else -1
    # Dénominateur géométrique du tenseur
    B_sq = 2*(C**2 + D**2) - sign * 2 * C * np.sqrt(C**2 + D**2)
    # Pulsation de résonance omega_*
    omega_star = (c_speed * np.sqrt(C**2 + D**2) / B_sq) * np.abs(m)
    return omega_star / (2 * np.pi) # En Hertz
```

#### Le Pipeline d'Observation Numérique :
1. **Acquisition :** Le vortex tourne en régime stationnaire. Le Pi déclenche le solénoïde qui vient frapper l'eau une seule fois à la périphérie. La caméra enregistre le "Ringdown" pendant 15 secondes.
2. **Reconstruction FCD :** L'algorithme `cv2.calcOpticalFlowFarneback` (OpenCV) extrait les déplacements de pixels et reconstruit $h(t, x, y)$.
3. **Séparation Azimutale :** Projection polaire et Transformée de Fourier spatiale pour isoler les harmoniques (les "bras" de la spirale, $m \in [-4, 4]$).
4. **Densité Spectrale (PSD) :** Transformée de Fourier temporelle sur chaque mode pour extraire le pic de résonance expérimental $f_{exp}(m)$.

#### L'Inversion Spectroscopique :
Vous fournissez au Pi vos $f_{exp}(m)$. Un algorithme des moindres carrés (`scipy.optimize.minimize`) cherche le couple $(C, D)$ qui annule l'erreur avec la fonction théorique. Le système déduit l'hydrodynamique sans jamais avoir inséré de vélocimètre.

**Contrôle Négatif Obligatoire (cf. Doc 1, Figure 1) :**
Biaisez volontairement l'algorithme en lui cachant les modes co-rotatifs ($m > 0$). L'optimiseur doit échouer à trouver un point unique et s'étaler dans une vallée d'erreur ("vortex homophoniques"). Cela prouve matériellement que la totalité de la frontière UV est requise pour figer le volume IR.

### 4. MANIP N — Le Canal Conforme 1D (Le Chef-d'œuvre Mathématique)
**Objectif :** Prouver que la perte d'information (les Greybody factors $R$ et $B$) peut être annulée en forçant une symétrie géométrique de jauge.

L'article de Coutant (Doc 3) montre que l'équation d'onde phononique correspond parfaitement à la théorie relativiste 1D (et devient invariante conforme) si et seulement si le produit $v(x) \cdot c(x)$ est constant le long du canal.

#### L'Ingénierie de la Symétrie :
- Dans votre canal, le débit volumique est $Q = v \cdot h \cdot b$ (où $b$ est la largeur, $h$ la profondeur).
- La vitesse des ondes est $c = \sqrt{gh}$.
- On impose $v(x) \cdot c(x) = K$.

En substituant $v$ : 
$$ \frac{Q}{h(x) \cdot b(x)} \sqrt{g \cdot h(x)} = K \implies \frac{1}{b(x) \sqrt{h(x)}} = \text{constante} $$

**La règle de construction :** Pour annuler l'écho de l'horizon, la largeur de votre canal $b(x)$ doit s'évaser proportionnellement à $1/\sqrt{h(x)}$. Vous n'allez donc pas simplement placer une bosse au fond du canal. Vous allez découper dans du plastique deux inserts latéraux qui élargissent le canal précisément là où l'eau devient moins profonde (sur la bosse).

#### Protocole Expérimental (Kymographe) :
- **Contrôle Négatif (Le bruit standard) :** Bosse au fond, mais canal droit ($b$ constant). Envoyez des ondes continues vers l'horizon. Le kymographe montrera une "pollution" : l'onde bloque, mais une forte réflexion repart en arrière (le facteur de corps gris $R \neq 0$).
- **La Preuve Conforme :** Placez vos inserts latéraux calculés. Refaites l'expérience. L'équation garantit que le couplage parasite s'effondre. Le kymographe montrera une onde qui s'écrase sur l'asymptote causale de manière absolue, absorbée dans un blueshift extrême, sans aucun écho. L'horizon est devenu "pur".

#### Conclusion
Ce LAB-3 est un tour de force épistémologique. Avec du plastique, du papier imprimé et un peu de code Python, vous ne faites plus "d'analogies pédagogiques". Vous construisez un calculateur analogique de 50 € capable de résoudre empiriquement des équations de gravité quantique semi-classique. Vous instanciez la théorie holographique dans la matière.

---
---

## [English] MEMO LAB-3: Analogue Gravity Cyber-Physical Observatory

Complete implementation of a Cyber-Physical Observatory for Analogue Gravity, moving from a brilliant idea to a genuine automated instrument.

### 1. The Technological Leap: Fast-Chequerboard Demodulation (FCD)
We discard the grazing laser from LAB-2. Papers mention a formidable, non-invasive, and free technique: **FCD (Fast-Chequerboard Demodulation)** or Synthetic Schlieren.

- **Principle:** A checkerboard pattern (2 mm squares) is placed under the transparent bottom of the tank, backlit uniformly. The Raspberry Pi camera films strictly from the zenith.
- **Physics:** The water surface acts as a field of microlenses. A height variation $\nabla h$ deflects the light and distorts the apparent checkerboard. An Optical Flow algorithm compares the distorted image to the resting image and reconstructs the 3D elevation map $\delta h(t, x, y)$ at 60 fps.

### 2. Nomenclature and Materials (BOM - Budget ~50 €)
The architecture separates metric creation (Pump) from quantum perturbation (Plucker) and measurement (FCD).

| Category | Component | Role | Cost |
| :--- | :--- | :--- | :--- |
| **2D Bulk** | Flat acrylic tank (e.g., 60x60 cm) with a center hole. | Hosts the Unruh Vortex. | ~15 € |
| **1D Bulk** | PVC gutter with profiled lateral inserts (see Manip N). | Hosts the Conformal Horizon. | ~10 € |
| **IR Generator** | Aquarium pump (1500 L/h) + Valves. | Sets the metric ($C$, $D$, Flow rate $Q$). | ~15 € |
| **UV Generator** | 5V Solenoid + plastic rod, driven by MOSFET. | The plucker (The black hole's "Pluck"). | ~5 € |
| **FCD Optics** | Printed checkerboard + LED panel (from an old LCD screen). | Optical reference target. | 0 € |
| **Brain** | Raspberry Pi 4 + Camera Module (HQ or V2). | Digital Twin and Interferometer. | Existing |

*(Plucker Wiring: The Pi drives the gate of an IRLZ44N MOSFET, which lets power flow to the solenoid to guarantee a temporally perfect 50 ms impact).*

### 3. MANIP M — Black Hole Spectroscopy (2D ABHS)
**Objective:** Validate state holography. Recover the flow geometry (Circulation $C$, Drain $D$) solely by "listening" to the surface waves.

#### The Digital Twin (`predict_abhs.py`):
The Pi computes the theoretical spectrum of shallow-water Light-Rings (Equation 9 from the first arXiv document).

```python
import numpy as np

def theoretical_light_ring_freq(m, C, D, h_water):
    c_speed = np.sqrt(9.81 * h_water)
    sign = 1 if m > 0 else -1
    # Geometric denominator of the tensor
    B_sq = 2*(C**2 + D**2) - sign * 2 * C * np.sqrt(C**2 + D**2)
    # Resonance pulsation omega_*
    omega_star = (c_speed * np.sqrt(C**2 + D**2) / B_sq) * np.abs(m)
    return omega_star / (2 * np.pi) # In Hertz
```

#### The Digital Observation Pipeline:
1. **Acquisition:** The vortex spins in a steady state. The Pi triggers the solenoid, which strikes the water once at the periphery. The camera records the "Ringdown" for 15 seconds.
2. **FCD Reconstruction:** The `cv2.calcOpticalFlowFarneback` algorithm (OpenCV) extracts pixel displacements and reconstructs $h(t, x, y)$.
3. **Azimuthal Separation:** Polar projection and spatial Fourier Transform to isolate the harmonics (the spiral "arms", $m \in [-4, 4]$).
4. **Power Spectral Density (PSD):** Temporal Fourier Transform on each mode to extract the experimental resonance peak $f_{exp}(m)$.

#### Spectroscopic Inversion:
You feed your $f_{exp}(m)$ to the Pi. A least-squares algorithm (`scipy.optimize.minimize`) seeks the $(C, D)$ pair that zeroes out the error against the theoretical function. The system deduces hydrodynamics without ever inserting a velocimeter.

**Mandatory Negative Control (cf. Doc 1, Figure 1):**
Intentionally bias the algorithm by hiding the co-rotating modes ($m > 0$) from it. The optimizer must fail to find a unique point and smear out in an error valley ("homophonic vortices"). This physically proves that the entire UV boundary is required to freeze the IR bulk.

### 4. MANIP N — The 1D Conformal Channel (The Mathematical Masterpiece)
**Objective:** Prove that information loss (the Greybody factors $R$ and $B$) can be cancelled by forcing a geometric gauge symmetry.

Coutant's paper (Doc 3) shows that the phononic wave equation matches 1D relativistic theory perfectly (and becomes conformally invariant) if and only if the product $v(x) \cdot c(x)$ is constant along the channel.

#### The Engineering of Symmetry:
- In your channel, the volumetric flow rate is $Q = v \cdot h \cdot b$ (where $b$ is the width, $h$ the depth).
- The wave speed is $c = \sqrt{gh}$.
- We impose $v(x) \cdot c(x) = K$.

Substituting $v$:
$$ \frac{Q}{h(x) \cdot b(x)} \sqrt{g \cdot h(x)} = K \implies \frac{1}{b(x) \sqrt{h(x)}} = \text{constant} $$

**The construction rule:** To cancel the horizon echo, your channel's width $b(x)$ must widen proportionally to $1/\sqrt{h(x)}$. Therefore, you will not simply place a bump at the bottom of the channel. You will cut two lateral inserts out of plastic that widen the channel precisely where the water gets shallower (over the bump).

#### Experimental Protocol (Kymograph):
- **Negative Control (Standard Noise):** Bump at the bottom, but a straight channel (constant $b$). Send continuous waves towards the horizon. The kymograph will show "pollution": the wave blocks, but a strong reflection bounces back (greybody factor $R \neq 0$).
- **The Conformal Proof:** Place your calculated lateral inserts. Repeat the experiment. The equation guarantees that the parasitic coupling collapses. The kymograph will show a wave crashing absolutely onto the causal asymptote, absorbed in an extreme blueshift, without any echo. The horizon has become "pure".

#### Conclusion
This LAB-3 is an epistemological tour de force. With plastic, printed paper, and some Python code, you are no longer making "pedagogical analogies." You are building a 50 € analogue computer capable of empirically solving semi-classical quantum gravity equations. You are instantiating holographic theory in matter.
