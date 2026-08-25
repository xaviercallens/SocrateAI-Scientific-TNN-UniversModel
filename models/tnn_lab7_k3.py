import torch
import torch.nn as nn

class K3PicardLatticeTNN(nn.Module):
    def __init__(self, input_mt_features=128, picard_rank=20):
        super(K3PicardLatticeTNN, self).__init__()
        # Maps Magnetotelluric crustal conductivity data to K3 Picard lattice parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_mt_features, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )
        self.lattice_decoder = nn.Linear(128, picard_rank)
        self.intersection_matrix = nn.Parameter(torch.randn(picard_rank, picard_rank))
        
    def forward(self, mt_data):
        # mt_data: (batch_size, input_mt_features)
        encoded = self.encoder(mt_data)
        picard_params = self.lattice_decoder(encoded) # (batch, picard_rank)
        
        # Symmetrize intersection matrix
        Q = self.intersection_matrix + self.intersection_matrix.t()
        
        # Calculate topological invariant (quadratic form)
        # H = p^T Q p
        p = picard_params.unsqueeze(-1) # (batch, 20, 1)
        p_t = picard_params.unsqueeze(1) # (batch, 1, 20)
        
        # Batch matrix multiplication
        topological_invariant = torch.bmm(torch.bmm(p_t, Q.unsqueeze(0).repeat(p.shape[0], 1, 1)), p).squeeze()
        return picard_params, topological_invariant

def train_and_save_lab7_model(save_path):
    print("[LAB-7 TNN] Initializing Telluric K3 Oracle TNN...")
    model = K3PicardLatticeTNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        dummy_mt_data = torch.randn(16, 128)
        picard_params, invariant = model(dummy_mt_data)
        loss = torch.mean(invariant**2) # Minimize deviation of invariant
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-7 TNN] Model saved to {save_path}")
