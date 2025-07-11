"""
neuron_selection_linear.py
==========================

Utilities to **extract**, **sample (random / fixed)**, and **test**
neuron-weight matrices for **Linear (fully-connected) layers only**.

The built-in smoke-test now uses a *tiny* all-Linear MLP instead of ResNet,
so you can run it without pulling in torchvision.
"""

from __future__ import annotations

import random
from typing import List

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1.  Extract weight matrices from Linear layers
# --------------------------------------------------------------------------- #
def get_neuron_list(model: nn.Module, *, select_layer_mode: str = "default") -> List[torch.Tensor]:
    """
    Return a list of weight tensors, one per `nn.Linear` layer in `model`.

    Args
    ----
    model
        Any PyTorch neural network.
    select_layer_mode
        * "default":  keep every Linear layer.
        * "filter" :  keep **odd-numbered** Linear layers only (1st, 3rd, …).

    Returns
    -------
    List[torch.Tensor]
        Each tensor has shape *(out_features, in_features)*.
    """
    neuron_list: List[torch.Tensor] = []
    layer_counter = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            layer_counter += 1

            if select_layer_mode == "filter" and layer_counter % 2 == 0:
                continue  # skip even layers

            neuron_list.append(module.weight)  # already (out, in)

    return neuron_list


# --------------------------------------------------------------------------- #
# 2.  Sampling helpers
# --------------------------------------------------------------------------- #
def _resolve_k(k: float | int, total: int) -> int:
    """Convert *k* (int or 0<k≤1) to a concrete integer within [1, total]."""
    if k <= 1.0:
        k = max(1, int(total * float(k)))
    return min(int(k), total)


def select_random_neurons(neuron_list: List[torch.Tensor], k: float | int = 12) -> List[torch.Tensor]:
    """Randomly select *k* neurons per layer."""
    selected: List[torch.Tensor] = []
    for w in neuron_list:
        n = w.shape[0]
        k_actual = _resolve_k(k, n)
        idx = random.sample(range(n), k_actual)
        selected.append(w[idx])
    return selected


def select_fixed_neurons(
    neuron_list: List[torch.Tensor],
    k: float | int = 12,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Deterministic selection using `torch.manual_seed(seed)`."""
    torch.manual_seed(seed)
    selected: List[torch.Tensor] = []
    for w in neuron_list:
        n = w.shape[0]
        k_actual = _resolve_k(k, n)
        idx = torch.randperm(n)[:k_actual]
        selected.append(w[idx])
    return selected


# --------------------------------------------------------------------------- #
# 3.  A tiny all-Linear network for smoke-testing
# --------------------------------------------------------------------------- #
class SimpleMLP(nn.Module):
    """
    Minimal MLP with three Linear layers.

    • Input features : 128
    • Hidden 1       : 64  (ReLU)
    • Hidden 2       : 32  (ReLU)
    • Output         : 10  (logits)
    """

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
        return self.net(x.view(x.size(0), -1))  # flatten


# --------------------------------------------------------------------------- #
# 4.  Simple smoke test
# --------------------------------------------------------------------------- #
def _test_neuron_functions() -> None:
    model = SimpleMLP()
    print("\nModel:", model.__class__.__name__)

    # --- extraction -------------------------------------------------------- #
    neurons_default = get_neuron_list(model, select_layer_mode="default")
    neurons_filtered = get_neuron_list(model, select_layer_mode="filter")

    print(f"Total Linear layers found: {len(neurons_default)}")
    print(f"Odd-numbered layers kept (filter mode): {len(neurons_filtered)}")

    # --- random sampling --------------------------------------------------- #
    subset = select_random_neurons(neurons_default, k=0.25)  # 25 % per layer
    print("Random sampling (25 % per layer):")
    print(
        "  original neurons:",
        sum(w.shape[0] for w in neurons_default),
        "→ selected:",
        sum(w.shape[0] for w in subset),
    )

    # --- fixed sampling ---------------------------------------------------- #
    sub1 = select_fixed_neurons(neurons_default, k=8, seed=123)
    sub2 = select_fixed_neurons(neurons_default, k=8, seed=123)
    identical = all(torch.equal(a, b) for a, b in zip(sub1, sub2))
    
    print("Fixed sampling (k=8) consistency:", "✓ identical" if identical else "✗ different")
    print(sum(w.shape[0] for w in sub1))


if __name__ == "__main__":
    _test_neuron_functions()
