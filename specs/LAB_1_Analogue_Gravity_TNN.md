# MEMO LAB-1: Literature-Sourced Protocols — Analogue Gravity in a Bathtub

**Statut :** Complément au MEMO LAB-0, basé sur une revue de littérature ciblée.
**Discipline :** Identique au reste du programme — pré-enregistrer la prédiction, mesurer, publier l'échec. Ce protocole teste explicitement le **Principe P4 (rebond / non-effondrement)** via un mécanisme d'horizon sonique.
**Banc de Validation IA :** Comparatif ResConv1D (résiduel) vs CNN baseline sur la dynamique des fluides inhomogènes.

---

## 1. La Gravité Analogique en Cuve (Analogue Gravity)

**Le Principe Physique :** Unruh (1981) a démontré qu'un écoulement fluide dont la vitesse dépasse localement la vitesse de phase des ondes crée un horizon : au-delà de ce point, une onde se propageant contre le courant ne peut plus remonter (structure causale identique à un trou noir). 
Le groupe Weinfurtner (Nottingham) réalise cela avec un vortex de vidange (draining vortex) — de l'eau s'échappant par le fond d'une cuve tout en tournant — créant un horizon sonique mesuré par profilométrie laser. Une variante plus simple (Rousseaux et al. 2008) utilise un canal d'écoulement avec un obstacle fuselé et un haut-parleur immergé injectant des ondes à contre-courant, bloquant la longue onde et la convertissant en une onde courte (signature classique de l'émission de Hawking stimulée).

**Pourquoi c'est pertinent pour le programme :** Ce n'est pas seulement une instance de P1 (borne), c'est une instance directe de **P4** : le mécanisme par lequel une région d'un système ondulatoire devient inaccessible car la vitesse de l'écoulement dépasse une vitesse caractéristique.

---

## 2. Protocole Low-Tech : Le Vortex de Vidange en Cuisine

**Matériel :**
*   Grande cuve transparente (boîte de rangement plastique $\approx 40\times30$ cm).
*   Trou de vidange au centre avec bouchon ajustable.
*   Pompe d'aquarium pour recirculer l'eau et maintenir un niveau stable.
*   Traceur de surface (farine légère ou paillettes).
*   Nappe laser (pointeur + tige en verre, cf. MANIP E du LAB-0).
*   Téléphone en slow-motion filmant d'en haut.

**Prédiction Pré-Enregistrée :**
La vitesse radiale $v(r)$ augmente vers le centre (conservation de la masse, $v(r) \sim 1/r$). Il doit exister un rayon $r_h$ où $v(r_h)$ égale la vitesse de propagation des ondes de surface en eau peu profonde $c \sim \sqrt{gh}$. 
Au-delà de $r_h$ vers le centre, un paquet d'ondes lancé depuis l'extérieur vers le centre ne doit plus pouvoir ressortir.

**Protocole d'Observation :**
Perturber la surface avec une goutte ou un léger tapotement à diverses distances du centre, filmer, et identifier la distance sous laquelle l'ondulation ne peut plus remonter le courant — c'est une mesure directe et visible de $r_h$, l'horizon sonique.

**Contrôle Négatif Obligatoire :** 
Arrêter la vidange (aucun écoulement radial). Aucune distance de non-retour ne doit apparaître ; les ondulations doivent se propager librement dans toutes les directions.

---

## 3. Avertissement Épistémologique (The Honesty Clause)

Un article fondamental de philosophie de la physique (Crowther et al., arXiv:1811.03859) est explicite : le système source (l'eau) et le système cible (un véritable trou noir gravitationnel) partagent une **équation**, pas une **physique**. L'analogie est purement mathématique (les deux obéissent à une équation de type Klein-Gordon sur une métrique effective) et ne permet aucune inférence sur la véritable gravité quantique.
**Règle absolue :** Vous observez un mécanisme d'horizon ondulatoire, *pas* un trou noir. Cette clause de rigueur s'applique de la même manière que notre quarantaine de la T-dualité dans le programme RAMA.

---

## 4. Benchmark IA (ResConv1D vs CNN) sur le Proxy de Vortex

Pour comparer les architectures sur ce problème, une simulation numérique 1D d'advection-diffusion scalaire a été exécutée sur fond d'écoulement divergent $v(r) = -A/r$. **Note d'honnêteté :** ce solveur est un modèle-jouet 1D inspiré du concept d'horizon, pas une reproduction des équations couplées hauteur/vitesse en eau peu profonde de Weinfurtner. Les deux réseaux apprennent à imiter un schéma aux différences finies déterministe et lisse — c'est un test d'interpolation de PDE, pas un test de compréhension physique de l'horizon.

**Protocole Numérique :**
*   Base d'entraînement : 2000 ondes, test sur 500 ondes.
*   Comparatif : CNN 1D baseline vs. ResConv1D (résiduel), **à budget de paramètres égalisé (~100k chacun)**.
*   **Contrôle négatif obligatoire :** répéter le même pipeline sur un champ sans horizon (Fr < 1 partout). Si l'écart CNN/ResConv1D persiste à l'identique, la mesure ne teste pas l'horizon, elle teste la capacité du modèle sur n'importe quel signal lisse.

**Ce que le benchmark mesure réellement :**
La dérive mesurée est la "L2-norm drift" (||u_pred||² vs ||u_target||²), pas un invariant hamiltonien — le système simulé contient un terme de diffusion artificielle qui le rend dissipatif par construction.

---

## 5. Références Bibliographiques Téléchargées & Intégrées

1. **Unruh, W. G. (1981).** *Experimental Black-Hole Evaporation?* Physical Review Letters, 46, 1351. [Le papier fondateur prouvant qu'un écoulement fluide transsonique modélise une métrique effective d'horizon].
2. **Weinfurtner, S., et al. (2011).** *Measurement of Stimulated Hawking Emission in an Analogue System.* Physical Review Letters, 106, 021302. [Observation expérimentale de la conversion de modes d'ondes dans un canal fluide].
3. **Rousseaux, G., et al. (2008).** *Observation of negative-frequency waves in a water tank: a classical analogue to the Hawking effect?* New Journal of Physics, 10, 053015. [L'expérience fondatrice du haut-parleur sous-marin à contre-courant].
4. ~~**Torres, T., et al. (2017).**~~ *Retiré* — Torres porte sur la superradiance rotationnelle (ergorégion) qui exige un terme azimutal absent de notre modèle 1D radial pur. Citer cette référence sans rotation dans le solveur constitue un décalage entre la caution littéraire et la simulation réelle.
5. **Crowther, K., et al. (2019).** *What we cannot learn from analogue experiments.* Synthese, 198(10), 3701-3726. (arXiv:1811.03859). [La clause d'honnêteté épistémologique : une équation partagée ne valide pas la gravité quantique].
6. **[Revue 2025]** *An introduction to nonlinear fiber optics and optical analogues to gravitational phenomena.* (arXiv:2512.15695). [Confirme que l'analogie en fibre optique requiert un équipement hors d'atteinte pour un labo de cave].
