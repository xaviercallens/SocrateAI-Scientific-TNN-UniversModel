import torch
import torch.nn as nn

class PersistentHomologyTNN(nn.Module):
    def __init__(self, point_dim=3, hidden_dim=64, barcode_dim=256):
        super(PersistentHomologyTNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, barcode_dim)
        )
        self.wasserstein_metric = nn.Linear(barcode_dim, 1)

    def forward(self, point_cloud):
        # point_cloud: (batch_size, num_points, 3)
        features = self.encoder(point_cloud)
        # Global max pooling (PointNet style)
        global_features, _ = torch.max(features, dim=1)
        # Estimate Wasserstein distance from a target Torus proxy
        wasserstein_dist = self.wasserstein_metric(global_features)
        return global_features, wasserstein_dist

def train_and_save_lab5_model(save_path):
    print("[LAB-5 TNN] Initializing Trans-Scale TDA Persistent Homology TNN...")
    model = PersistentHomologyTNN()
    # Dummy training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        dummy_points = torch.randn(8, 100, 3)
        _, w_dist = model(dummy_points)
        loss = w_dist.mean()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(f"[LAB-5 TNN] Model saved to {save_path}")
