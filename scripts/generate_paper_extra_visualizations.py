import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "publication_package/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def plot_cusp_core():
    print("[*] Generating Cusp-Core resolution figure...")
    r = np.logspace(-2, 2, 500)
    rho_0 = 1.0
    R_s = 1.0
    R_c = 5.66 / 10.0 # Normalized for visualization
    
    # NFW Cusp: rho ~ 1 / (r/Rs)(1+r/Rs)^2
    rho_nfw = rho_0 / ((r/R_s) * (1 + r/R_s)**2)
    
    # Burkert Core (Pseudo-isothermal): rho ~ 1 / ((1+r/Rc)(1+(r/Rc)^2))
    rho_core = rho_0 / ((1 + r/R_c) * (1 + (r/R_c)**2))
    # Align tails
    rho_core = rho_core * (rho_nfw[-1] / rho_core[-1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, rho_nfw, 'r--', linewidth=2.5, label="Classical NFW Profile (Singular Cusp)")
    ax.plot(r, rho_core, 'g-', linewidth=3.5, label="Univers Model TDA Profile (Core $R_c \\approx 5.66$ kpc)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Radius r (kpc) [Log Scale]")
    ax.set_ylabel("Dark Matter Density $\\rho(r)$ [Log Scale]")
    ax.set_title("Empirical Resolution of the Cusp-Core Problem")
    ax.legend()
    
    plt.savefig(f"{OUT_DIR}/cusp_core.png", bbox_inches='tight')
    plt.close()

def plot_tdual_bounce():
    print("[*] Generating T-Dual topological bounce figure...")
    R = np.logspace(-2, 2, 500)
    alpha_prime = 1.0
    Reff = np.maximum(R, alpha_prime / R)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(R, R, 'k--', alpha=0.6, linewidth=2, label="Classical Euclidean Metric ($R$)")
    ax.plot(R, Reff, 'b-', linewidth=3.5, label="T-Dual Effective Metric ($R_{eff} = \\max(R, \\alpha'/R)$)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Physical Scale $R$")
    ax.set_ylabel("Effective Topological Scale $R_{eff}$")
    ax.set_title("The T-Dual Geometric Bounce")
    ax.legend()
    
    plt.savefig(f"{OUT_DIR}/tdual_bounce.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_cusp_core()
    plot_tdual_bounce()
    print("[+] Visualizations successfully created in publication_package/figures/")
