# MEMO LAB-0: Cave Laboratory — Salvage-Hardware Experiments & TNN Validation

**Statut :** mémo de conception expérimentale. Aucune de ces manips ne valide une théorie ; chacune **instancie un principe dans un cas particulier**, avec une prédiction exacte connue d'avance et un contrôle négatif obligatoire.
**Discipline :** identique au reste du programme — pré-enregistrer la prédiction, mesurer, publier l'échec. Un carnet séparé `LAB_LL.md` recense les manips ratées.
**Principes testés :** P1 (borne auto-duale), P3 (le discret épingle le continu), P5 (quantification par fermeture), la cascade (contexte Navier–Stokes), et la **comparaison de modèles neuronaux (ResConv1D vs CNN/MLP baseline)** sur des simulations physiques.

---

## 0. Sécurité — à lire avant d'acheter ou de démonter quoi que ce soit

1. **Lasers.** Les diodes de lecteurs CD/DVD/Blu-ray récupérées sont souvent **non collimatées et bien plus puissantes** que les pointeurs du commerce ; celles de graveurs peuvent dépasser 100 mW. Jamais d'œil dans le faisceau, jamais de réflexion spéculaire non contrôlée (retirer montre et bague), lunettes adaptées à la longueur d'onde, faisceau toujours **sous le niveau des yeux**, mur mat en fond de course. Un laser de graveur DVD peut brûler la rétine avant le réflexe palpébral.
2. **Secteur.** Toute alimentation récupérée reste sur bloc externe basse tension. Les alimentations d'ordinateur contiennent des condensateurs chargés après débranchement : ne pas ouvrir.
3. **Cave = humidité.** Tout le montage sur différentiel 30 mA, rien au sol, électronique sur étagère, pas de rallonge en flaque.
4. **Interdits formels.** Cryogénie (hélium, azote liquide) : asphyxie et surpression, hors de portée d'un labo domestique. Haute tension. Lasers de classe 4 non protégés. Ces trois lignes ne sont pas négociables quel que soit le budget.
5. **Bruit acoustique.** Au-delà de ~85 dB prolongé, protection auditive ; les ultrasons ne sont pas inaudibles pour tout le monde et fatiguent.

---

## 1. MANIP A — Le réseau CD/DVD : P3 avec un étalon gratuit

**Principe testé :** P3 — un invariant discret (le pas du réseau) épingle une observable continue (la longueur d'onde).
**Matériel :** un CD et un DVD sacrifiés, pointeur laser, mètre ruban, mur.
**Prédiction exacte pré-enregistrée :** `d·sin θ = m·λ`. Avec d = 1,6 µm (CD) et λ ≈ 650 nm, l'ordre 1 sort à θ ≈ 24°. Avec le DVD (d = 0,74 µm), θ ≈ 61°.
**Protocole :** mesurer la distance écran L et l'écart des ordres x, en déduire λ ; comparer à la valeur nominale du laser.
**Contrôle négatif obligatoire :** un disque **vierge non gravé** ou un CD-R dont la couche est intacte doit donner un motif différent ; un miroir plan ne doit donner **aucun** ordre latéral.

**Comparaison IA (ResConv1D vs MLP Baseline) :**
Si l'on entraîne une IA à prédire l'angle $\theta$ en fonction de $\lambda$ et $d$ :
*   **Modèle Traditionnel (MLP) :** Devra apprendre la fonction sinus par cœur via des milliers de paramètres, avec une forte dérive (MSE élevée) hors de son domaine de données d'entraînement.
*   **ResConv1D (résiduel) :** En imposant un biais inductif via la structure résiduelle, le modèle résiduel apprend plus efficacement la relation — mais cet écart reste à valider à budget de paramètres égal.

---

## 2. MANIP B — La fente variable : P1 mesuré à la main

**Principe testé :** P1 — produit conservé, borne auto-duale.
**Matériel :** deux lames de rasoir montées sur une charnière ou deux vis, laser, écran.
**Prédiction pré-enregistrée :** `a · Δx ≈ constante` — la largeur de la figure varie comme l'inverse de la largeur de fente. Tracer log(Δx) contre log(a) donne une droite de pente −1.
**Protocole :** cinq à huit largeurs de fente, photo à distance fixe, mesure de l'écart des minima sur l'image (ImageJ).
**Contrôle négatif :** un trou circulaire donne des anneaux d'Airy, pas des franges linéaires.

---

## 3. MANIP C — Diffraction acousto-optique dans l'eau (la plus belle)

**Principe testé :** P1 + P3 + P5 simultanément — une onde stationnaire crée un réseau discret qui contraint la lumière.
**Matériel :** cuve transparente, transducteur piézo de récupération (ex: humidificateur ~1,7 MHz, ~10 €), générateur = carte son ou Pi Pico + ampli, laser, écran.
**Prédiction pré-enregistrée :** l'espacement du réseau est `Λ = c_eau/f` avec c_eau ≈ 1480 m/s. À 1,7 MHz, Λ ≈ 0,87 mm.
**Contrôle négatif :** couper l'excitation → le motif doit disparaître complètement. Désaccorder hors résonance → l'amplitude doit s'effondrer.

**Comparaison IA (ResConv1D vs CNN Baseline) :**
Simuler la propagation couplée acoustique-optique :
*   **Modèle Traditionnel (CNN 2D) :** Souffrira d'atténuation numérique et peinera à résoudre l'interférence en champ lointain.
*   **ResConv1D (résiduel) :** La structure résiduelle avec GroupNorm tend à mieux capturer les motifs périodiques. La dérive de norme L2 est attendue plus faible. Note : l'énergie n'est *pas* une quantité conservée dans ce système dissipatif — la métrique mesure l'écart au schéma numérique de référence, pas une conservation hamiltonienne.

---

## 4. MANIP D — Cavité acoustique : P5 sous sa forme la plus nue

**Principe testé :** P5 — la fermeture quantifie.
**Matériel :** tube PVC (longueur mesurée), haut-parleur à une extrémité, micro électret, balayage de fréquence piloté par le Pi.
**Prédiction pré-enregistrée :** tube ouvert-fermé, `f_n = (2n−1)·c/(4L)` — donc **harmoniques impaires uniquement**.
**Contrôle négatif :** boucher une extrémité doit **faire disparaître les harmoniques paires**.

---

## 5. MANIP E — Cascade turbulente en cuisine (contexte Navier–Stokes)

**Principe testé :** la cascade elle-même, pour l'intuition.
**Matériel :** bac d'eau, traceurs (poivre moulu), téléphone en slow-motion, nappe laser. Analyse : **PIVlab** (gratuit, MATLAB/Octave).
**Prédiction :** spectre d'énergie en −5/3 de Kolmogorov.
**Contrôle :** eau au repos → spectre plat, aucune structure.

---

## 6. MANIP F — Ising sur le Raspberry : le point auto-dual, prédiction kernel-checked

**Principe testé :** P1 — le point fixe de la dualité localise le point critique.
**Matériel :** le Pi seul (ou n'importe quel ordinateur).
**Prédiction exacte (prouvée en Lean) :** `K_c = log(1+√2)/2 ≈ 0,44069`.
**Protocole :** Metropolis Monte Carlo sur grilles 16², 32², 64² ; pic de susceptibilité.
**Contrôle négatif :** le même code en 1D ne doit montrer **aucune** transition à température finie.

---

## 7. MANIP G — Comparaison Numérique : Équation de Burgers & Chocs

**Principe testé :** Comparaison à budget de paramètres égal entre un MLP baseline et un ResConv1D (résiduel) sur l'imitation d'un schéma aux différences finies pour l'équation de Burgers 1D.
**Matériel :** Raspberry Pi 4/5 ou PC local, script de profiling Python (`torch.autograd.profiler`).
**Ce que la manip mesure réellement :** Les deux réseaux apprennent à imiter un solveur déterministe aux différences finies. L'équation de Burgers contient un terme de viscosité qui rend le système dissipatif par construction — il n'y a donc pas de conservation hamiltonienne au sens strict. La métrique "L2-norm drift" mesure l'écart entre la norme L2 de la prédiction et celle de la cible dissipative.

**Protocole d'Expérimentation :**
1. Générer une marche d'onde de choc 1D (Burgers déterministe).
2. Lancer l'inférence sur le **MLP baseline** et mesurer MSE et dérive de norme L2 sur 100 timesteps.
3. Lancer l'inférence sur le **ResConv1D (résiduel)** à budget de paramètres comparable, mêmes métriques.
4. **Contrôle négatif obligatoire :** Répéter sur un champ sans choc (viscosité dominante, pas de discontinuité). Si l'écart persiste à l'identique, la mesure teste la capacité du modèle sur n'importe quel signal lisse, pas le choc lui-même.

**Résultats Attendus :**
| Modèle | MSE | Dérive Norme L2 | Paramètres |
| :--- | :--- | :--- | :--- |
| **MLP Baseline** | À mesurer | À mesurer | ~100k |
| **ResConv1D (résiduel)** | À mesurer | À mesurer | ~100k |

**Note d'honnêteté :** Un écart de MSE entre deux architectures à paramètres inégaux ne démontre rien sur l'induction physique. Le gain doit être isolé par ablation (connexions résiduelles ? normalisation ? profondeur ?) pour être interprétable.

---

## 8. Ordre recommandé et budget

| Ordre | Manip | Coût | Risque d'échec |
|---|---|---|---|
| 1 | F (Ising) & G (vHPU TNN) | 0 € | nul (Purement Computationnel) |
| 2 | A (réseau CD/DVD) | ~5 € | faible |
| 3 | D (cavité acoustique) | ~15 € | faible |
| 4 | B (fente variable) | ~5 € | moyen (mécanique fine) |
| 5 | E (cascade) | ~10 € | moyen (analyse bruitée) |
| 6 | C (acousto-optique) | ~30 € | élevé — repli Chladni prévu |

**Total : moins de 70 €**, l'essentiel en récupération (et logiciel Open-Source).
