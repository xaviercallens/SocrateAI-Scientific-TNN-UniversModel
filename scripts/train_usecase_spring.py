import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# 1. GÉNÉRATION DE DONNÉES : Oscillateur Harmonique à 2 Corps (Ressort)
# =============================================================================
def get_spring_derivatives(x, k=1.0, m=1.0):
    # x : [batch, 8] -> (q1x, q1y, q2x, q2y, p1x, p1y, p2x, p2y)
    q1, q2, p1, p2 = torch.split(x, 2, dim=1)
    
    # dq/dt = p / m
    dq1 = p1 / m
    dq2 = p2 / m
    
    # dp/dt = Force
    dp1 = -k * (q1 - q2)
    dp2 = k * (q1 - q2)
    
    return torch.cat([dq1, dq2, dp1, dp2], dim=1)

def generate_dataset(num_samples=5000):
    # Positions et Momentums initiaux aléatoires
    x = torch.randn(num_samples, 8) * 1.5
    
    # Les cibles sont les dérivées temporelles exactes (la vraie physique)
    dx_dt = get_spring_derivatives(x)
    return x, dx_dt

# =============================================================================
# 2. MODÈLE TNN THERMODYNAMIQUE (HNN)
# =============================================================================
class SimpleHNN(nn.Module):
    """
    Un Hamiltonien Neural Network simplifié pour apprendre la physique du ressort.
    Prend x = (q, p) et prédit un scalaire H (l'Énergie).
    """
    def __init__(self, in_dim=8, hidden_dim=128):
        super().__init__()
        # Le réseau apprend le Hamiltonien H(q, p)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(), # Tanh est souvent préféré pour l'Autograd des HNN (courbure lisse)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Matrice Symplectique M (Levi-Civita)
        M = torch.eye(in_dim)
        M = torch.cat([M[in_dim//2:], -M[:in_dim//2]])
        self.register_buffer('M', M)
        
    def forward(self, x):
        # x doit requérir le gradient !
        H = self.net(x)
        
        # dH/dx
        dH_dx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        
        # Les équations de Hamilton : dx/dt = dH/dx @ M^T
        dx_dt = dH_dx @ self.M.t()
        return dx_dt, H

# =============================================================================
# 3. ENTRAÎNEMENT ET VALIDATION
# =============================================================================
def train():
    print("--- Génération du Dataset Physique (Ressort 2-Corps 2D) ---")
    x_train, dx_train = generate_dataset(5000)
    x_test, dx_test = generate_dataset(1000)
    
    model = SimpleHNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    epochs = 400
    batch_size = 256
    
    print("\n--- Début de l'entraînement du Moteur Thermodynamique ---")
    for epoch in range(epochs):
        permutation = torch.randperm(x_train.size()[0])
        total_loss = 0.
        
        for i in range(0, x_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_dx = x_train[indices], dx_train[indices]
            
            # Nécessaire pour autograd
            batch_x.requires_grad_(True)
            
            optimizer.zero_grad()
            
            pred_dx, _ = model(batch_x)
            loss = loss_fn(pred_dx, batch_dx)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 50 == 0 or epoch == epochs - 1:
            # Évaluation
            x_test.requires_grad_(True)
            pred_test_dx, _ = model(x_test)
            test_loss = loss_fn(pred_test_dx, dx_test).item()
            print(f"Epoch {epoch:03d} | Train Loss: {total_loss/batch_size:.4e} | Test Loss: {test_loss:.4e}")
            
    return model

def evaluate_energy_conservation(model):
    print("\n--- Validation : Conservation de l'Énergie ---")
    
    # 1. Créer un état initial pour le système (t=0)
    x_t = torch.randn(1, 8)
    
    dt = 0.01
    steps = 500
    
    energies = []
    
    # 2. Simuler la trajectoire
    for _ in range(steps):
        x_t.requires_grad_(True)
        dx_dt, H = model(x_t)
        
        energies.append(H.item())
        
        # Intégration d'Euler basique (Euler symplectique ou RK4 serait mieux)
        # On détache pour éviter de construire un graphe géant
        with torch.no_grad():
            x_t = x_t + dx_dt * dt
            
    energy_drift = max(energies) - min(energies)
    print(f"Énergie Initiale : {energies[0]:.4f}")
    print(f"Énergie Finale   : {energies[-1]:.4f}")
    print(f"Dérive Max (Drift) : {energy_drift:.4e}")
    
    if energy_drift < 0.1:
        print("✅ PASS : Le réseau a appris les lois de la thermodynamique (Conservation).")
    else:
        print("⚠️ WARNING : La dérive est un peu haute. Un intégrateur RK4 améliorerait la stabilité.")

if __name__ == "__main__":
    trained_model = train()
    evaluate_energy_conservation(trained_model)
