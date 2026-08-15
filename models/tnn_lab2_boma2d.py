import torch
import torch.nn as nn

class BOMA2DPivTNN(nn.Module):
    """
    LAB-2 TNN: Neural PIV / BOMA-2D Dual-Axis Stepper Decoder.
    Maps 32x32 photodiode intensity grid to velocity vector fields (u, v).
    """
    def __init__(self, grid_size=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1) # 2 channels for u, v velocity components
        )

    def forward(self, intensity_map):
        if intensity_map.dim() == 2:
            intensity_map = intensity_map.unsqueeze(0).unsqueeze(0)
        elif intensity_map.dim() == 3:
            intensity_map = intensity_map.unsqueeze(1)
            
        velocity_field = self.encoder(intensity_map)
        return velocity_field

def train_and_save_lab2_model(save_path="models/tnn_lab2_boma2d.pt"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = BOMA2DPivTNN(32)
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-2 TNN] BOMA-2D Neural PIV Model initialized & saved to: {save_path}")

if __name__ == "__main__":
    train_and_save_lab2_model()
