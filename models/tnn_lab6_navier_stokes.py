import torch
import torch.nn as nn
import math

class LerayHopfNavierStokesTNN(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32):
        super(LerayHopfNavierStokesTNN, self).__init__()
        # A simple CNN for predicting next step of vector field
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def leray_hopf_projection(self, v_field):
        # v_field: (batch, 2, H, W)
        # Simplified Leray-Hopf Projection in Fourier space
        # F(P u) = F(u) - k (k . F(u)) / |k|^2
        B, C, H, W = v_field.shape
        v_ft = torch.fft.fftn(v_field, dim=[-2, -1])
        
        ky = torch.fft.fftfreq(H).view(-1, 1).repeat(1, W).to(v_field.device)
        kx = torch.fft.fftfreq(W).view(1, -1).repeat(H, 1).to(v_field.device)
        
        k_sq = kx**2 + ky**2
        k_sq[0, 0] = 1.0 # Avoid division by zero
        
        # Dot product k . F(u)
        k_dot_v = kx * v_ft[:, 0, :, :] + ky * v_ft[:, 1, :, :]
        
        # Projection
        v_ft_proj = torch.zeros_like(v_ft)
        v_ft_proj[:, 0, :, :] = v_ft[:, 0, :, :] - kx * k_dot_v / k_sq
        v_ft_proj[:, 1, :, :] = v_ft[:, 1, :, :] - ky * k_dot_v / k_sq
        
        # Inverse FFT
        v_proj = torch.fft.ifftn(v_ft_proj, dim=[-2, -1]).real
        return v_proj

    def forward(self, x):
        # Predict next step
        next_v = self.conv_net(x) + x
        # Project to divergence-free field
        div_free_v = self.leray_hopf_projection(next_v)
        return div_free_v

def train_and_save_lab6_model(save_path):
    print("[LAB-6 TNN] Initializing Leray-Hopf Z3 Navier-Stokes TNN...")
    model = LerayHopfNavierStokesTNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        dummy_field = torch.randn(2, 2, 32, 32)
        out_field = model(dummy_field)
        loss = torch.mean(out_field**2) # Dummy loss on enstrophy proxy
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-6 TNN] Model saved to {save_path}")
