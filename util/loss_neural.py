# contrastive_linear_loss.py
# ==========================
#
# Cosine-similarity **hinge loss** for *Linear* layer neurons (row-vectors).
#
#   hinge(i,j) = max(0,  cosine_sim(w_i , w_j) − margin)
#
# Loss is averaged over positive hinges **per layer**, then across all layers.
# Chunking keeps GPU memory under control.

from __future__ import annotations
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1.  Cached upper-triangular indices (shared across instances)
# --------------------------------------------------------------------------- #
class _IdxCache:
    _cache: dict[tuple[int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def get(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (n, device.type, device.index or 0)
        if key not in _IdxCache._cache:
            _IdxCache._cache[key] = torch.triu_indices(
                n, n, 1, device=device, dtype=torch.long
            )
        return _IdxCache._cache[key]


# --------------------------------------------------------------------------- #
# 2.  Main loss  (cosine only)
# --------------------------------------------------------------------------- #
class ContrastiveLinearLoss(nn.Module):
    """
    Cosine-hinge loss for neuron weight vectors of **Linear** layers.

    Parameters
    ----------
    margin : float
        Penalise similarities greater than this value (0 < margin < 1).
    pair_batch : int
        Maximum number of (i,j) pairs processed in one chunk (memory knob).
    """

    def __init__(self, margin: float = 0.9, pair_batch: int = 600_000) -> None:
        super().__init__()
        assert 0.0 < margin < 1.0, "margin must be in (0,1)"
        self.margin = margin
        self.pair_batch = max(1, pair_batch)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _pairwise_cosine(
        self, w: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor
    ) -> torch.Tensor:
        v1, v2 = w[i_idx], w[j_idx]          # (pairs, D)
        return F.cosine_similarity(v1, v2, dim=1)

    # ------------------------------------------------------------------ #
    def _layer_hinge(
        self, w: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute hinge loss for all (i,j) pairs of a single layer (chunked)."""
        hinge_sum, count_pos = 0.0, 0

        for start in range(0, i_idx.numel(), self.pair_batch):
            s = slice(start, start + self.pair_batch)
            sim = self._pairwise_cosine(w, i_idx[s], j_idx[s])
            hinge = F.relu(sim - self.margin)         # similarity > margin
            pos = hinge > 0
            hinge_sum += hinge[pos].sum()
            count_pos += pos.sum().item()

        return hinge_sum / max(count_pos, 1)

    # ------------------------------------------------------------------ #
    def forward(self, weight_list: List[torch.Tensor]) -> torch.Tensor:
        """
        weight_list – list of Linear weight tensors, each (N, D) with N = neurons.
        """
        device, total, n_layers = weight_list[0].device, 0.0, 0

        for w in weight_list:
            assert w.ndim == 2, "Expected 2-D weight matrix (out_features × in_features)"
            n = w.shape[0]
            i_idx, j_idx = _IdxCache.get(n, device)
            total += self._layer_hinge(w, i_idx, j_idx)
            n_layers += 1

        return total / max(n_layers, 1)


# --------------------------------------------------------------------------- #
# 3.  Tiny all-Linear network for demos
# --------------------------------------------------------------------------- #
class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# --------------------------------------------------------------------------- #
# 4.  Smoke / stress tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # helper
    def run_loss(loss_fn: nn.Module, w: torch.Tensor, label: str) -> None:
        t0 = time.time()
        val = loss_fn([w.to(device)])
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        print(f"{label:<25} loss = {val.item():.6f}  ({1000*(time.time()-t0):.1f} ms)")

    # ------------------------------------------------------------------ #
    # 1. Small demo
    # ------------------------------------------------------------------ #
    N_small, D = 64, 128
    w_small = torch.randn(N_small, D, device=device)
    loss_fn = ContrastiveLinearLoss(margin=0.1).to(device)

    print("\n--- SMALL DEMO (64×128) ---")
    run_loss(loss_fn, w_small, "random")

    # ------------------------------------------------------------------ #
    # 2. Similar vs. different
    # ------------------------------------------------------------------ #
    print("\n--- SIMILAR / DIFFERENT ---")
    base = torch.randn(1, D, device=device)
    w_similar = base + 0.01 * torch.randn(N_small, D, device=device)
    w_diff = 2.0 * torch.randn(N_small, D, device=device)

    run_loss(loss_fn, w_similar, "similar")
    run_loss(loss_fn, w_diff, "different")

    # ------------------------------------------------------------------ #
    # 3. Big stress test
    # ------------------------------------------------------------------ #
    print("\n--- BIG STRESS TEST (4096×256) ---")
    N_big, D_big = 4096, 256
    w_big = 0.05 * torch.randn(N_big, D_big, device=device)
    run_loss(loss_fn, w_big, "big random")

    # ------------------------------------------------------------------ #
    # 4. Model integration
    # ------------------------------------------------------------------ #
    print("\n--- MODEL INTEGRATION ---")
    model = SimpleMLP(in_features=D).to(device)
    weight_list = [p.detach() for p in model.parameters() if p.ndim == 2]
    model_loss = loss_fn(weight_list)
    print(f"SimpleMLP total loss: {model_loss.item():.6f}")
