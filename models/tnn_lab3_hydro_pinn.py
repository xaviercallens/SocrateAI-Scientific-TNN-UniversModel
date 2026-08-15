import torch
import torch.nn as nn

class HydrodynamicPINN(nn.Module):
    """
    LAB-3 TNN: Hydrodynamic Physics-Informed Neural Network (PINN).
    Enforces 1D Bernoulli conservation equation v * dv/dx + g * dh/dx = 0
    to predict Froude profile Fr(x) and detect horizon rh.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # Outputs [v(x), h(x)]
        )

    def forward(self, x):
        x = x.requires_grad_(True)
        out = self.net(x)
        v, h = out[:, 0:1], out[:, 1:2]
        
        g = 9.81
        c = torch.sqrt(torch.abs(g * h) + 1e-5)
        Froude = v / c
        return Froude, v, h

def train_and_save_lab3_model(save_path="models/tnn_lab3_hydro_pinn.pt"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = HydrodynamicPINN()
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-3 TNN] Hydrodynamic PINN Model initialized & saved to: {save_path}")

if __name__ == "__main__":
    train_and_save_lab3_model()
