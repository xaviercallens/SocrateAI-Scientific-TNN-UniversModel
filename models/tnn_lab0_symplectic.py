import torch
import torch.nn as nn
import numpy as np

class SymplecticTNN(nn.Module):
    """
    LAB-0 TNN: Symplectic Hamiltonian Neural Network.
    Preserves phase-space (q, p) invariants for baseline digital twin predictions.
    """
    def __init__(self, state_dim=2, hidden_dim=64):
        super().__init__()
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Symplectic matrix J = [[0, 1], [-1, 0]]
        self.register_buffer("J", torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))

    def forward(self, state):
        state = state.requires_grad_(True)
        H = self.hamiltonian_net(state)
        dH = torch.autograd.grad(H.sum(), state, create_graph=True)[0]
        # dq/dt = dH/dp, dp/dt = -dH/dq -> dState/dt = dH * J^T
        state_dot = torch.matmul(dH, self.J.T)
        return state_dot, H

def train_and_save_lab0_model(save_path="models/tnn_lab0_symplectic.pt"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = SymplecticTNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Synthetic Harmonic Oscillator dataset (q, p)
    t = torch.linspace(0, 10, 100)
    q = torch.cos(t)
    p = -torch.sin(t)
    states = torch.stack([q, p], dim=1)
    
    for epoch in range(100):
        optimizer.zero_grad()
        dot, H = model(states)
        loss = torch.mean(dot**2) + torch.std(H) # Enforce energy conservation
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-0 TNN] Symplectic Model trained & saved to: {save_path}")

if __name__ == "__main__":
    train_and_save_lab0_model()
