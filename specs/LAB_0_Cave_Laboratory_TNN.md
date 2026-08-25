# MEMO LAB-0: Cave Laboratory — Salvage-Hardware Experiments on the Dual-Scale Principles

**Statut :** mémo de conception expérimentale. Aucune de ces manips ne valide une
théorie ; chacune **instancie un principe dans un cas particulier**, avec une
prédiction exacte connue d'avance et un contrôle négatif obligatoire.
**Discipline :** identique au reste du programme — pré-enregistrer la prédiction,
mesurer, publier l'échec. Un carnet séparé `LAB_LL.md` recense les manips ratées.
**Principes testés :** P1 (borne auto-duale), P3 (le discret épingle le continu),
P5 (quantification par fermeture), et la cascade (contexte Navier–Stokes).

---

## 0. Sécurité — à lire avant d'acheter ou de démonter quoi que ce soit

1. **Lasers.** Les diodes de lecteurs CD/DVD/Blu-ray récupérées sont souvent
   **non collimatées et bien plus puissantes** que les pointeurs du commerce ;
   celles de graveurs peuvent dépasser 100 mW. Jamais d'œil dans le faisceau,
   jamais de réflexion spéculaire non contrôlée (retirer montre et bague),
   lunettes adaptées à la longueur d'onde, faisceau toujours **sous le niveau des
   yeux**, mur mat en fond de course. Un laser de graveur DVD peut brûler la
   rétine avant le réflexe palpébral.
2. **Secteur.** Toute alimentation récupérée reste sur bloc externe basse tension.
   Les alimentations d'ordinateur contiennent des condensateurs chargés après
   débranchement : ne pas ouvrir.
3. **Cave = humidité.** Tout le montage sur différentiel 30 mA, rien au sol,
   électronique sur étagère, pas de rallonge en flaque.
4. **Interdits formels.** Cryogénie (hélium, azote liquide) : asphyxie et
   surpression, hors de portée d'un labo domestique. Haute tension. Lasers de
   classe 4 non protégés. Ces trois lignes ne sont pas négociables quel que soit
   le budget.
5. **Bruit acoustique.** Au-delà de ~85 dB prolongé, protection auditive ; les
   ultrasons ne sont pas inaudibles pour tout le monde et fatiguent.

---

## 1. Réponses directes à vos questions matériel

**Fibres optiques ?** Oui, mais pour ce qu'elles font bien. Une fibre plastique
(POF, câble Toslink audio de récupération, ~1 mm) est parfaite comme **guide et
comme ligne à retard** : c'est un système fermé, donc un support naturel de P5.
Ce qu'elle ne fera pas facilement : de l'interférométrie propre (les fibres
monomodes verre exigent clivage, alignement micrométrique, stabilité thermique —
faisable mais c'est un projet en soi, pas une première manip).

**Émetteurs acoustiques ?** Oui, et c'est votre meilleur rapport
qualité/prix/sécurité. Un haut-parleur récupéré + une carte son (ou la sortie
casque) pilotée en générateur de fréquence, un micro électret ou un micro USB :
vous avez un banc de spectroscopie complet pour ~10 €. Les modes propres d'un
tube ou d'une cavité sont P5 dans sa forme la plus pure.

**Diffraction dans un liquide ?** Oui — et c'est la manip la plus élégante du
mémo (§4) : une onde acoustique stationnaire dans l'eau crée un **réseau de
diffraction d'indice** (effet Debye–Sears / acousto-optique), et le laser
diffracte dessus. Vous fabriquez le réseau discret vous-même, et vous en réglez
le pas par la fréquence. P1 et P3 dans une seule cuve.

**Raspberry Pi ?** Oui, mais choisissez le bon rôle. Le Pi est excellent pour
piloter, journaliser, horodater et publier des certificats de mesure (et il fait
tourner votre discipline `.meta`/hash). Pour l'acquisition analogique rapide, sa
faiblesse est l'absence d'ADC intégré : ajoutez un ADS1115 (lent, précis, I²C,
~5 €) pour le continu, ou un MCP3008 pour le rapide-modeste. Pour l'audio, la
carte son USB est meilleure que tout GPIO. Un Pi Pico (~5 €) est en réalité plus
adapté pour l'échantillonnage temps-réel régulier, avec le Pi comme cerveau.

**Vieux lecteur CD/DVD ?** C'est la pièce maîtresse de la récupération, et pour
trois raisons distinctes : (a) le **disque lui-même** est un réseau de
diffraction de pas connu — 1,6 µm pour un CD, 0,74 µm pour un DVD — donc un
étalon gratuit ; (b) la **diode laser** (≈650 nm DVD, ≈780 nm CD) ; (c) le
**bloc optique et son moteur pas-à-pas**, qui est une platine de translation
micrométrique déjà assemblée — la pièce la plus chère d'un labo optique, gratuite
dans une déchetterie.

---

## 2. MANIP A — Le réseau CD/DVD : P3 avec un étalon gratuit

**Principe testé :** P3 — un invariant discret (le pas du réseau) épingle une
observable continue (la longueur d'onde).
**Matériel :** un CD et un DVD sacrifiés, pointeur laser, mètre ruban, mur.
**Prédiction exacte pré-enregistrée :** `d·sin θ = m·λ`. Avec d = 1,6 µm (CD) et
λ ≈ 650 nm, l'ordre 1 sort à θ ≈ 24°. Avec le DVD (d = 0,74 µm), θ ≈ 61°.
**Protocole :** mesurer la distance écran L et l'écart des ordres x, en déduire
λ ; comparer à la valeur nominale du laser.
**Contrôle négatif obligatoire :** un disque **vierge non gravé** ou un CD-R dont
la couche est intacte doit donner un motif différent ; un miroir plan ne doit
donner **aucun** ordre latéral. Si vous voyez des ordres sur le miroir, votre
montage diffracte ailleurs.
**Précision attendue :** quelques %. **Ce que ça enseigne :** que le pas discret
détermine exactement la géométrie continue — et vous aurez calibré votre laser,
utile pour toute la suite.

## 3. MANIP B — La fente variable : P1 mesuré à la main

**Principe testé :** P1 — produit conservé, borne auto-duale.
**Matériel :** deux lames de rasoir montées sur une charnière ou deux vis (la
platine du lecteur CD fait un excellent micro-déplacement), laser, écran.
**Prédiction pré-enregistrée :** `a · Δx ≈ constante` — la largeur de la figure
varie comme l'inverse de la largeur de fente. Tracer log(Δx) contre log(a) doit
donner une droite de pente −1.
**Protocole :** cinq à huit largeurs de fente, photo à distance fixe, mesure de
l'écart des minima sur l'image (ImageJ, gratuit).
**Contrôle négatif :** un trou circulaire donne des anneaux d'Airy, pas des
franges linéaires — le motif doit changer de nature, pas seulement d'échelle.
**Ce que ça enseigne :** vous mesurez de vos mains le principe d'incertitude
géométrique, c'est-à-dire l'ancêtre optique de `Reff_ge_sqrt`.

## 4. MANIP C — Diffraction acousto-optique dans l'eau (la plus belle)

**Principe testé :** P1 + P3 + P5 simultanément — une onde stationnaire (donc
fermée, donc quantifiée) crée un réseau discret qui contraint la lumière.
**Matériel :** cuve transparente (aquarium, boîte plastique), transducteur
piézo de récupération (buzzer, nettoyeur à ultrasons bon marché ~20 €, ou
haut-parleur exciter collé sur la paroi), générateur = carte son ou Pi Pico +
ampli, laser, écran.
**Prédiction pré-enregistrée :** l'espacement du réseau est `Λ = c_eau/f` avec
c_eau ≈ 1480 m/s. À 40 kHz, Λ ≈ 37 mm — trop grand pour diffracter du visible ;
**c'est pourquoi la manip exige des MHz**, donc un transducteur de nettoyeur à
ultrasons dans le meilleur des cas ne suffira pas et il faudra viser
l'ordre du MHz (transducteur d'humidificateur à brouillard, ~1,7 MHz, ~10 €, qui
donne Λ ≈ 0,87 mm — encore grand mais on entre dans le régime observable en
champ lointain avec un montage soigné).
**Statut honnête :** c'est la manip la plus incertaine du mémo. **Repli garanti :**
même sans diffraction visible, l'onde stationnaire elle-même est observable —
figures de Chladni sur une plaque, ou motif de nœuds visible à la surface du
liquide, ce qui teste P5 seul et fonctionne à coup sûr.
**Contrôle négatif :** couper l'excitation → le motif doit disparaître
complètement. Désaccorder hors résonance → l'amplitude doit s'effondrer.

## 5. MANIP D — Cavité acoustique : P5 sous sa forme la plus nue

**Principe testé :** P5 — la fermeture quantifie.
**Matériel :** tube PVC (longueur mesurée), haut-parleur à une extrémité, micro
électret ou téléphone, balayage de fréquence piloté par le Pi.
**Prédiction pré-enregistrée :** tube ouvert-ouvert, `f_n = n·c/(2L)` ; tube
ouvert-fermé, `f_n = (2n−1)·c/(4L)` — donc **harmoniques impaires uniquement**.
C'est la prédiction la plus discriminante du mémo.
**Protocole :** balayage 100 Hz–4 kHz, FFT, relevé des pics, ajustement de la
suite d'entiers ; correction de bout (≈0,6 × rayon) à appliquer.
**Contrôle négatif :** boucher une extrémité doit **faire disparaître les
harmoniques paires**. Si elles restent, votre fermeture fuit — et c'est
exactement le point : la quantification dépend de la condition aux limites, pas
du milieu.
**Extension fibre optique (votre question) :** la même logique dans un anneau de
fibre POF fermé sur lui-même avec une LED modulée et une photodiode donne une
ligne à retard ; les résonances de boucle sont P5 en optique. Plus délicat,
à tenter après D.

## 6. MANIP E — Cascade turbulente en cuisine (contexte Navier–Stokes)

**Principe testé :** la cascade elle-même, pour l'intuition.
**Matériel :** bac d'eau, traceurs (poivre moulu, paillettes, particules de
polystyrène), téléphone en slow-motion, éclairage rasant (une nappe laser
s'obtient en passant le faisceau sur une tige de verre ou un cylindre
transparent). Analyse : **PIVlab** (gratuit, MATLAB/Octave).
**Prédiction :** spectre d'énergie en −5/3 de Kolmogorov sur une décade, au
mieux deux. **Attente honnête : c'est peu, et le bruit dominera facilement.**
**Variante supérieure :** film de savon vertical ou grande bulle — turbulence
quasi-2D, où la cascade **s'inverse** (l'énergie remonte vers les grandes
échelles). Voir la dimensionnalité changer la physique de la cascade est
précisément l'intuition que votre modèle dyadique manipule abstraitement.
**Contrôle :** eau au repos → spectre plat, aucune structure.

## 7. MANIP F — Ising sur le Raspberry : le point auto-dual, prédiction kernel-checked

**Principe testé :** P1 — le point fixe de la dualité localise le point critique.
**Matériel :** le Pi seul (ou n'importe quel ordinateur).
**Prédiction exacte, déjà prouvée en Lean** (`kramers_wannier_self_dual`) :
`K_c = log(1+√2)/2 ≈ 0,44069`.
**Protocole :** Metropolis Monte Carlo sur grilles 16², 32², 64² ; pic de
susceptibilité ; extrapolation en taille finie.
**Contrôle négatif :** le même code en 1D ne doit montrer **aucune** transition à
température finie.
**Pourquoi je la place ici malgré son caractère numérique :** c'est la seule
manip du mémo dont la prédiction est démontrée par le noyau de Lean dans votre
propre dépôt. Boucle complète théorie → vérification formelle → expérience.

---

## 8. Instrumentation commune (le rôle du Raspberry)

- **Acquisition :** carte son USB pour l'audio (meilleure que tout GPIO) ;
  ADS1115 (I²C, ~5 €) pour le lent et précis ; Pi Pico pour l'échantillonnage
  temps-réel régulier, le Pi comme cerveau et journal.
- **Optique :** module caméra Pi ou vieux webcam ; l'exposition manuelle est
  indispensable (une exposition automatique ruine toute photométrie).
- **Mécanique :** le bloc optique de lecteur CD/DVD fournit moteur pas-à-pas,
  vis sans fin et glissières — une platine de translation gratuite, pilotable
  par driver A4988 (~2 €).
- **Discipline programme appliquée au labo :** chaque run écrit un `.meta`
  (date, matériel, réglages, hash des données brutes) ; les prédictions sont
  commitées **avant** la mesure ; les manips ratées vont dans `LAB_LL.md`.

## 9. Ordre recommandé et budget

| Ordre | Manip | Coût | Risque d'échec |
|---|---|---|---|
| 1 | F (Ising) | 0 € | nul |
| 2 | A (réseau CD/DVD) | ~5 € | faible |
| 3 | D (cavité acoustique) | ~15 € | faible |
| 4 | B (fente variable) | ~5 € | moyen (mécanique fine) |
| 5 | E (cascade) | ~10 € | moyen (analyse bruitée) |
| 6 | C (acousto-optique) | ~30 € | élevé — repli Chladni prévu |

**Total : moins de 70 €**, l'essentiel en récupération.

## 10. Ce que ces manips ne feront pas

Elles ne testent aucune conjecture du programme. Elles instancient des principes
**déjà établis**, dans des cas particuliers, avec des prédictions connues. Leur
valeur est l'intuition, la calibration du jugement, et la vérification que le
langage commun (P1, P3, P5) décrit bien des choses qui existent. C'est
exactement ce que vous en attendiez, et c'est une raison suffisante.
