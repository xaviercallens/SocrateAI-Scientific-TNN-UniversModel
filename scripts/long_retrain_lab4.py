import torch
import torch.nn as nn
import numpy as np
import os
import time

from tnn_holographic_p4_detector import HolographicTNN

def long_retrain_holographic_tnn(duration_seconds=3600, save_path="models/tnn_lab4_holographic_p4_retrained_1h.pt"):
    """
    Retrains & fine-tunes HolographicTNN (chi=8) on the Shallow Water CFD dataset
    for a specified duration (e.g., 1 hour).
    """
    dataset_path = "data/lab4_datasets/shallow_water_cfd.pt"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset {dataset_path} not found.")
        return

    print(f"\n[LONG RETRAINING] Starting {duration_seconds/3600:.1f}-hour continuous fine-tuning on Open Datasets...")
    dataset = torch.load(dataset_path, weights_only=False)
    boundary_tensor = dataset["boundary"]
    bulk_tensor = dataset["bulk"]
    
    batch_size, boundary_dim = boundary_tensor.shape
    _, bulk_res, _ = bulk_tensor.shape
    bond_dim = 8
    
    model = HolographicTNN(boundary_dim=boundary_dim, bond_dimension_chi=bond_dim, bulk_resolution=bulk_res)
    
    # Try loading existing weights to continue training
    prev_weights = "models/tnn_lab4_holographic_p4_retrained.pt"
    if os.path.exists(prev_weights):
        model.load_state_dict(torch.load(prev_weights, weights_only=False))
        print("[LONG RETRAINING] Loaded previous fine-tuned weights for continuous training.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Lower learning rate for fine-tuning
    criterion = nn.MSELoss()
    
    model.train()
    start_time = time.time()
    epoch = 0
    best_loss = float('inf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > duration_seconds:
                print(f"\n[LONG RETRAINING] Target duration of {duration_seconds} seconds reached. Stopping.")
                break
                
            optimizer.zero_grad()
            
            # Add slight data augmentation (noise) to prevent overfitting during long runs
            noisy_boundary = boundary_tensor + 0.01 * torch.randn_like(boundary_tensor)
            
            predicted_bulk = model(noisy_boundary)
            loss = criterion(predicted_bulk, bulk_tensor)
            loss.backward()
            optimizer.step()
            
            epoch += 1
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                # Save checkpoint
                torch.save(model.state_dict(), save_path)
            
            if epoch % 100 == 0:
                print(f" -> Elapsed: {elapsed_time/60:.2f} min / {duration_seconds/60:.2f} min | Epoch {epoch} | Best Loss MSE: {best_loss:.6f}")
                
            # Sleep slightly to simulate heavier batch computation / prevent full CPU lockup
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n[LONG RETRAINING] Interrupted by user.")

    print(f"[LONG RETRAINING COMPLETED] Final fine-tuned model weights saved to: {save_path}")

if __name__ == "__main__":
    # We set it to 3600 seconds (1 hour) for the real execution
    # Note: When run as a background task, it will continue running.
    long_retrain_holographic_tnn(duration_seconds=3600)
