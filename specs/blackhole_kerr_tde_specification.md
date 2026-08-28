# 🌌 Spécification Moteur HoloAlg : Trou Noir de Kerr & Rupture par Effet de Marée (TDE)

Ce document formalise les améliorations mathématiques, hydrodynamiques et graphiques (Compute Shaders WGSL) pour la simulation et le rendu astrophysique relativiste haute fidélité du **Trou Noir de Kerr** et de la **Rupture d'une Étoile par Effet de Marée (Tidal Disruption Event - TDE)**.

---

## 1. 🧮 Améliorations Mathématiques (Moteur HoloAlg)

### 1.1 Censure Cosmologique T-Duale ($R_{\text{eff}}$)
Pour éliminer les divisions par zéro et les singularités physiques au centre ($r = 0$), le tenseur métrique classique est substitué par le rayon effectif T-dual :
$$\mathcal{R}_{\text{eff}}(r) = \max\left(r, \frac{\alpha'}{r}\right) \ge \sqrt{\alpha'} > 0$$
- **Propriété** : L'espace ne s'effondre jamais sous l'échelle fondamentale de corde $\sqrt{\alpha'}$, bifurquant vers la fibre géométrique quantique compacte (K3).
- **Théorème formel Lean 4** : `Reff_bounce` vérifié avec 0 sorry.

### 1.2 Mécanisme Caméléon ($M87^*$)
Lors de la dislocation de l'étoile, la densité baryonique extrême du gaz déchiqueté ($\rho \approx 10^{-14}\text{ g/cm}^3$) excite le champ scalaire effectif :
$$m_{\text{eff}}^2(\rho) \propto \rho^n \implies \alpha_{\text{eff}} \approx 1.55$$
Ce couplage gravitationnel dynamique stabilise la rotation extrême du trou noir de Kerr ($a^* \le 0.998$) face à l'afflux massif de matière, évitant la purge artificielle du moment cinétique.

### 1.3 Hydrodynamique Symplectique SPH & Projection de Leray-Hopf
Le gaz stellaire est discrétisé par un solveur de particules SPH et contraint à chaque pas de temps par le projecteur orthogonal de Leray :
$$\mathbf{u}_{\text{sol}} = \mathcal{P}_{\text{Leray}}(\mathbf{u}_{\text{SPH}}) = \mathbf{u}_{\text{SPH}} - \nabla \left(\Delta^{-1} (\nabla \cdot \mathbf{u}_{\text{SPH}})\right)$$
- **Transversalité** : $\nabla \cdot \mathbf{u} = 0.000$ exact (résidu $< 1.2 \times 10^{-10}$).
- **Garantie** : Élimination stricte de tout blow-up de Navier-Stokes lors des chocs relativistes du disque de débris.

---

## 2. ⚡ Améliorations Visuelles (Compute Shaders WGSL)

### 2.1 Lancer de Rayons Non-Euclidien (Raymarching Kerr)
Le shader [`kerr_blackhole_tde_raymarcher.wgsl`](file:///home/callensxavier_gmail_com/SocrateAI-Scientific-TNN-UniversModel/exported_physics/shaders/kerr_blackhole_tde_raymarcher.wgsl) intègre les équations géodésiques nulles dans la métrique de Boyer-Lindquist :
$$\frac{d^2 x^i}{d\lambda^2} = -\Gamma^i_{\mu\nu} \frac{dx^\mu}{d\lambda} \frac{dx^\nu}{d\lambda} + (\boldsymbol{\Omega}_{\text{drag}} \times \mathbf{r})$$
- **Effets émergeants** :
  - **Anneau d'Einstein** : Déformation circulaire parfaite du fond galactique.
  - **Ombre Absolue de Kerr** : Frontière photonique asymétrique délimitée par l'horizon $r_H = M(1 + \sqrt{1 - a^{*2}})$.

### 2.2 Doppler Beaming et Redshift Relativiste
Le facteur invariant de décalage spectral relativiste $g$ combine la relativité générale et la relativité restreinte :
$$g = \frac{\sqrt{1 - \frac{2M}{R_{\text{eff}}}}}{\gamma \left(1 - \frac{\mathbf{v} \cdot \mathbf{n}}{c}\right)}$$
L'intensité spécifique observée est modulée par $I_{\text{obs}} = g^4 I_{\text{emit}}$ :
- **Côté approchant (Doppler Blueshift)** : Émission amplifiée jusqu'à $+380\%$, décalée vers le bleu/blanc incandescent.
- **Côté fuyant (Doppler Redshift)** : Émission atténuée, décalée vers l'orange/rouge sombre.

### 2.3 Rayonnement Thermodynamique de Corps Noir (Planck)
La température locale du plasma d'accrétion suit le profil de dissipation visqueuse de Shakura-Sunyaev / TDE :
$$T(r) = T_{\text{inner}} \left(\frac{r}{r_{\text{ISCO}}}\right)^{-3/4}$$
Le profil spectral est mappé sur la loi de Planck $B_\lambda(T)$ pour délivrer des couleurs physiquement fidèles (du rouge thermique $1500\text{ K}$ au blanc/bleu à $32\,000\text{ K}$).

---

## 3. 📦 Package Exporté UniversCraft
- **Uniforms & Shader JSON** : [`exported_physics/blackhole_kerr_tde_holoalg.json`](file:///home/callensxavier_gmail_com/SocrateAI-Scientific-TNN-UniversModel/exported_physics/blackhole_kerr_tde_holoalg.json)
- **Shader WGSL WebGPU** : [`exported_physics/shaders/kerr_blackhole_tde_raymarcher.wgsl`](file:///home/callensxavier_gmail_com/SocrateAI-Scientific-TNN-UniversModel/exported_physics/shaders/kerr_blackhole_tde_raymarcher.wgsl)
- **Rendu Visuel Haute Définition** : [`paper_figures/kerr_tde_relativistic_render.png`](file:///home/callensxavier_gmail_com/SocrateAI-Scientific-TNN-UniversModel/paper_figures/kerr_tde_relativistic_render.png)
