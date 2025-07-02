import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

# Import kornia for fast GPU SSIM
from kornia.metrics import ssim as kornia_ssim


def _ssim_pair_gpu(k1: torch.Tensor, k2: torch.Tensor, win_size: int | None = None) -> float:
    """
    Fast GPU-based SSIM between two single-channel (d × d) kernels using kornia.
    """
    # Ensure tensors are on the same device and have the right shape
    device = k1.device
    k1 = k1.to(device).float()
    k2 = k2.to(device).float()
    
    # Kornia SSIM expects 4D tensors: (B, C, H, W)
    # Our kernels are 2D: (H, W), so we need to add batch and channel dims
    k1_4d = k1.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    k2_4d = k2.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Determine window size
    h, w = k1.shape
    if win_size is None:
        win_size = min(h, w) if min(h, w) % 2 == 1 else min(h, w) - 1
        win_size = min(win_size, 11)
    
    # Ensure odd window size
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)  # Minimum window size of 3
    
    # Calculate dynamic range
    data_range = float((torch.max(torch.maximum(k1, k2)) - torch.min(torch.minimum(k1, k2))).item())
    if data_range == 0:
        data_range = 1.0
    
    # Use kornia SSIM and extract the mean value
    ssim_val = kornia_ssim(k1_4d, k2_4d, window_size=win_size, max_val=data_range)
    
    # Handle different possible return shapes from kornia SSIM
    if ssim_val.numel() == 1:
        return float(ssim_val.item())
    else:
        # Take the mean if SSIM returns a spatial map
        return float(ssim_val.mean().item())


class ContrastiveKernelLoss(nn.Module):
    r"""
    GPU-accelerated Contrastive-Kernel SSIM hinge loss.

        hinge_ij = max(0, SSIM(k_i, k_j) - margin)

    For each layer we average **only the positive hinge terms** (pairs whose
    similarity exceeds `margin`); pairs with SSIM ≤ margin do not contribute.
    The final loss is the mean of the per-layer losses.
    """

    def __init__(self, margin: float = 0.90, win_size: int | None = None):
        """
        margin   – similarity above this value is penalised (0 < margin < 1).
        win_size – odd SSIM window size. If None, use the largest odd value
                   ≤11 that fits inside each kernel.
        """
        super().__init__()
        self.margin = margin
        self.win_size = win_size

    # ------------------------------------------------------------------ #
    # internal helper: pairwise SSIM
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pairwise_ssim(kernels: torch.Tensor, win_size: int) -> torch.Tensor:
        """
        Return the SSIM for every unordered pair (i < j) of kernels.

        kernels : (n, h, w)
        win_size: odd window size for SSIM
        returns : (n·(n-1)/2,) vector of SSIM values
        """
        device = kernels.device
        n, h, w = kernels.shape

        # force odd window size ≥3
        win_size = max(3, win_size if win_size % 2 == 1 else win_size - 1)

        # dynamic range (avoid zero)
        data_range = float((kernels.max() - kernels.min()).item() or 1.0)

        # indices of upper-triangular pairs
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)

        # gather pairs and compute SSIM
        ssim_vals = kornia_ssim(
            kernels[i_idx].unsqueeze(1),  # (pairs,1,h,w)
            kernels[j_idx].unsqueeze(1),  # (pairs,1,h,w)
            window_size=win_size,
            max_val=data_range,
        )

        # kornia can return extra dims; flatten them
        if ssim_vals.ndim > 1:
            ssim_vals = ssim_vals.mean(dim=tuple(range(1, ssim_vals.ndim)))

        return ssim_vals  # (pairs,)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, kernels_list: list[torch.Tensor]) -> torch.Tensor:
        """
        kernels_list – list of tensors, each of shape (n_k, d, d)
        """
        device = kernels_list[0].device
        total_loss, num_layers = 0.0, 0

        for kernels in kernels_list:
            n, d, d2 = kernels.shape
            assert d == d2, "Each kernel must be square (d × d)"

            # choose SSIM window
            w = self.win_size
            if w is None:
                w = d if d % 2 == 1 else d - 1
                w = min(w, 11)

            # pairwise SSIM → hinge → keep positives only
            ssim_vals = self._pairwise_ssim(kernels, w)
            hinge_vals = F.relu(ssim_vals - self.margin)  # (pairs,)

            active = hinge_vals > 0
            if active.any():
                layer_loss = hinge_vals[active].mean()
            else:
                layer_loss = torch.tensor(0.0, device=device)

            total_loss += layer_loss
            num_layers += 1

        return total_loss / num_layers if num_layers else torch.tensor(0.0, device=device)


# A simple model with 3 convolutional layers.
class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # For simplicity, each conv has one input channel so that weight shape is (out_channels, 1, k, k)
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3, padding=1)  # kernels: (512, 1, 3, 3)
        self.conv2 = nn.Conv2d(1, 512, kernel_size=7, padding=3)  # kernels: (512, 1, 7, 7)
        self.conv3 = nn.Conv2d(1, 512, kernel_size=5, padding=2)   # kernels: (512, 1, 5, 5)
        
        # Initialize with similar patterns
        self._init_similar_weights()

    def _init_similar_weights(self):
        """Initialize convolutional layers with similar patterns for testing."""
        with torch.no_grad():
            # Base pattern for 3x3 kernels (Gaussian-like)
            base_3x3 = torch.tensor([
                [0.1, 0.2, 0.1],
                [0.2, 0.4, 0.2],
                [0.1, 0.2, 0.1]
            ], dtype=torch.float32)
            
            # Initialize conv1 (3x3) with similar patterns
            for i in range(self.conv1.weight.size(0)):
                noise = torch.randn_like(base_3x3) * 0.05  # Small random variation
                self.conv1.weight[i, 0] = base_3x3 + noise
            
            # Base pattern for larger kernels (edge-like patterns)
            base_edge = torch.tensor([
                [-0.1, -0.1, 0.0, 0.1, 0.1],
                [-0.1, -0.2, 0.0, 0.2, 0.1],
                [-0.1, -0.2, 0.0, 0.2, 0.1],
                [-0.1, -0.2, 0.0, 0.2, 0.1],
                [-0.1, -0.1, 0.0, 0.1, 0.1]
            ], dtype=torch.float32)
            
            # Initialize conv3 (5x5) with similar edge patterns
            for i in range(self.conv3.weight.size(0)):
                noise = torch.randn_like(base_edge) * 0.1
                self.conv3.weight[i, 0] = base_edge + noise
            
            # For 7x7 kernels, create expanded version of edge pattern
            base_7x7 = torch.zeros(7, 7, dtype=torch.float32)
            base_7x7[1:6, 1:6] = base_edge
            
            # Initialize conv2 (7x7) with similar patterns
            for i in range(self.conv2.weight.size(0)):
                noise = torch.randn_like(base_7x7) * 0.02
                self.conv2.weight[i, 0] = base_7x7 + noise

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

import random



if __name__ == "__main__":
    # Instantiate the model.
    model = SimpleConvModel()
    
    # Extract kernels from each convolution.
    # Since the conv layers have weight shape (out_channels, 1, k, k), we squeeze the channel dimension.
    kernels1 = model.conv1.weight.squeeze(1)  # shape: (512, 3, 3)
    kernels2 = model.conv2.weight.squeeze(1)  # shape: (512, 7, 7)
    kernels3 = model.conv3.weight.squeeze(1)  # shape: (512, 5, 5)
    
    # Put all conv kernels into a list.
    kernels_list = [kernels1, kernels2, kernels3]
    
    # Test GPU-accelerated implementation
    print("Testing GPU-accelerated ContrastiveKernelSSIMLoss:")
    loss_fn = ContrastiveKernelLoss(margin=0.1)
    loss_value = loss_fn(kernels_list)
    print("Contrastive Kernel Loss:", loss_value.item())
    
    print("\n" + "="*50)
    print("Testing with two similar kernels")
    print("="*50)
    
    # Create two similar 3x3 kernels for testing
    kernel1 = torch.tensor([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0], 
        [1.0, 2.0, 1.0]
    ], dtype=torch.float32)
    
    # Create a similar kernel with slight variation
    kernel2 = torch.tensor([
        [1.1, 2.1, 0.9],
        [1.9, 4.1, 2.1],
        [0.9, 1.9, 1.1]  
    ], dtype=torch.float32)
    
    # Stack them to create a tensor with 2 kernels
    similar_kernels = torch.stack([kernel1, kernel2], dim=0)  # shape: (2, 3, 3)
    
    print(f"Kernel 1:\n{kernel1}")
    print(f"Kernel 2:\n{kernel2}")
    
    # Test SSIM directly between the two kernels
    ssim_value = _ssim_pair_gpu(kernel1, kernel2, win_size=3)
    print(f"\nDirect SSIM between similar kernels: {ssim_value:.4f}")
    
    # Test with different margins
    print(f"\nTesting different margins (win_size=3):")
    for margin in [0.5, 0.7, 0.9, 0.95]:
        loss_fn_margin = ContrastiveKernelLoss(margin=margin, win_size=3)
        loss_margin = loss_fn_margin([similar_kernels])
        print(f"Margin={margin}: Loss={loss_margin.item():.6f}")
