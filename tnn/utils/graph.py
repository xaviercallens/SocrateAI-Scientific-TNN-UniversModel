"""
TNN Graph Utilities
====================
Shared graph construction functions for PyTorch Geometric batching.
Eliminates duplicated `build_fc_edges` / `build_fully_connected_edges`
across multiple scripts.
"""
import torch


def build_fully_connected_edges(num_atoms: int, batch_size: int = 1, device=None):
    """
    Construct a fully-connected edge_index for N atoms, batched.
    
    Full connectivity avoids human-biased pairwise pruning of
    N-body interactions in the graph representation.
    
    Args:
        num_atoms: Number of nodes per system/molecule
        batch_size: Number of systems in the batch
        device: Target torch device
    
    Returns:
        edge_index: [2, num_edges] LongTensor
    """
    edges = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edges.append([i, j])

    if not edges:
        result = torch.empty((2, 0), dtype=torch.long)
        return result.to(device) if device else result

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Replicate for batch
    edge_index_batch = []
    for b in range(batch_size):
        edge_index_batch.append(edge_index + b * num_atoms)

    result = torch.cat(edge_index_batch, dim=1)
    return result.to(device) if device else result
