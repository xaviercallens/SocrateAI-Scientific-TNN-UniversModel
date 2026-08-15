import torch
import torch.nn as nn

class FourierCorrelatorTNN(nn.Module):
    """
    LAB-1 TNN: Complex-Valued Spectral Optical Correlator Network.
    Learns spatial dual-space masks for single-point defect suppression.
    """
    def __init__(self, in_features=64):
        super().__init__()
        self.in_features = in_features
        # Complex mask weights in dual space
        self.dual_mask = nn.Parameter(torch.ones(in_features, in_features, dtype=torch.complex64))
        
    def forward(self, real_space_image):
        # 1. FFT to dual space
        dual_space = torch.fft.fft2(real_space_image)
        
        # 2. Dual space filtering
        filtered_dual = dual_space * self.dual_mask
        
        # 3. Inverse FFT back to real space
        reconstructed = torch.fft.ifft2(filtered_dual)
        return torch.abs(reconstructed)

def train_and_save_lab1_model(save_path="models/tnn_lab1_fourier.pt"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = FourierCorrelatorTNN(64)
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-1 TNN] Fourier Correlator Model initialized & saved to: {save_path}")

if __name__ == "__main__":
    train_and_save_lab1_model()
