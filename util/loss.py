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
        kernels_list – list of tensors, each of shape (n_k, d, d) or (in_ch, n_k, d, d) for channel diversity
        """
        device = kernels_list[0].device
        total_loss, num_layers = 0.0, 0

        for kernels in kernels_list:
            if len(kernels.shape) == 4:  # Channel diversity mode: (in_ch, out_ch, h, w)
                in_ch, out_ch, d, d2 = kernels.shape
                assert d == d2, "Each kernel must be square (d × d)"

                # choose SSIM window
                w = self.win_size
                if w is None:
                    w = d if d % 2 == 1 else d - 1
                    w = min(w, 11)

                # Calculate loss for each input channel separately
                channel_losses = []
                for ch in range(in_ch):
                    ch_kernels = kernels[ch]  # shape: (out_ch, h, w)
                    
                    # Skip if only one kernel in this channel
                    if out_ch <= 1:
                        continue
                    
                    # pairwise SSIM → hinge → keep positives only
                    ssim_vals = self._pairwise_ssim(ch_kernels, w)
                    hinge_vals = F.relu(ssim_vals - self.margin)  # (pairs,)

                    active = hinge_vals > 0
                    if active.any():
                        ch_loss = hinge_vals[active].mean()
                        channel_losses.append(ch_loss)

                # Average across input channels
                if channel_losses:
                    layer_loss = torch.stack(channel_losses).mean()
                else:
                    layer_loss = torch.tensor(0.0, device=device)

            else:  # Normal mode: (n_k, d, d)
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
    for margin in [0.1,0.5, 0.7, 0.9, 0.95]:
        loss_fn_margin = ContrastiveKernelLoss(margin=margin, win_size=3)
        loss_margin = loss_fn_margin([similar_kernels])
        print(f"Margin={margin}: Loss={loss_margin.item():.6f}")
    
    print("\n" + "="*60)
    print("TESTING CHANNEL DIVERSITY MODE")
    print("="*60)
    
    # Test channel diversity mode with TWO CASES: similar vs different kernels
    print("Testing channel diversity mode with ContrastiveKernelLoss:")
    
    # Create test kernels in channel diversity format: (in_ch, out_ch, h, w)
    # Simulate 3 input channels, 4 output channels per input channel, 3x3 kernels
    in_channels, out_channels, kernel_size = 3, 4, 3
    
    # CASE 1: Similar kernels within each input channel
    print("\n--- CASE 1: Similar kernels (should produce HIGH loss with low margin) ---")
    similar_kernels = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    
    # Base patterns for each input channel
    base_patterns = [
        torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]),  # Gaussian-like
        torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]),  # Vertical edge
        torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])   # Horizontal edge
    ]
    
    # Fill each input channel with similar kernels
    for in_ch in range(in_channels):
        base_pattern = base_patterns[in_ch]
        for out_ch in range(out_channels):
            # Add small random noise to create similar but not identical kernels
            noise = torch.randn_like(base_pattern) * 0.1
            similar_kernels[in_ch, out_ch] = base_pattern + noise
    
    print(f"Created similar kernels with shape: {similar_kernels.shape}")
    
    # CASE 2: Different kernels within each input channel
    print("\n--- CASE 2: Different kernels (should produce LOW loss) ---")
    different_kernels = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    
    # Create very different patterns for each kernel within each input channel
    different_patterns = [
        # Input channel 0: Very different patterns
        [
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # Top-left corner
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),  # Center
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # Bottom-right corner
            torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])  # Diagonal
        ],
        # Input channel 1: More different patterns
        [
            torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # Top row
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),  # Middle row
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # Bottom row
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   # Left column
        ],
        # Input channel 2: Even more different patterns
        [
            torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]),  # Checkerboard
            torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),      # Inverted checkerboard
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),      # Diagonal line
            torch.tensor([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]])       # Anti-diagonal line
        ]
    ]
    
    # Fill each input channel with different kernels
    for in_ch in range(in_channels):
        for out_ch in range(out_channels):
            different_kernels[in_ch, out_ch] = different_patterns[in_ch][out_ch]
    
    print(f"Created different kernels with shape: {different_kernels.shape}")
    
    # Test both cases with same margins
    print(f"\nTesting both cases with margin=0.1:")
    loss_fn_cd = ContrastiveKernelLoss(margin=0.1, win_size=3)
    
    loss_similar = loss_fn_cd([similar_kernels])
    loss_different = loss_fn_cd([different_kernels])
    
    print(f"Similar kernels loss (margin=0.1): {loss_similar.item():.6f}")
    print(f"Different kernels loss (margin=0.1): {loss_different.item():.6f}")
    print(f"Difference: {loss_similar.item() - loss_different.item():.6f}")
    
    # Test with different margins for both cases
    print("\nTesting different margins for both cases:")
    for margin in [0.1, 0.3, 0.5, 0.7, 0.9]:
        loss_fn_cd_margin = ContrastiveKernelLoss(margin=margin, win_size=3)
        loss_similar_margin = loss_fn_cd_margin([similar_kernels])
        loss_different_margin = loss_fn_cd_margin([different_kernels])
        print(f"Margin={margin}: Similar={loss_similar_margin.item():.6f}, Different={loss_different_margin.item():.6f}")
    
    # Test with multiple layers in channel diversity mode
    print("\nTesting multiple layers in channel diversity mode:")
    # Create another layer with different dimensions
    layer2_similar = torch.zeros(2, 6, 5, 5)  # 2 input channels, 6 output channels, 5x5 kernels
    layer2_different = torch.zeros(2, 6, 5, 5)
    
    # Fill with similar patterns for layer2_similar
    for in_ch in range(2):
        base_5x5 = torch.randn(5, 5) * 0.5
        for out_ch in range(6):
            noise = torch.randn_like(base_5x5) * 0.1
            layer2_similar[in_ch, out_ch] = base_5x5 + noise
    
    # Fill with different patterns for layer2_different
    for in_ch in range(2):
        for out_ch in range(6):
            # Create very different random patterns
            layer2_different[in_ch, out_ch] = torch.randn(5, 5) * 2.0
    
    # Test with both layers
    multi_layer_loss_similar = loss_fn_cd([similar_kernels, layer2_similar])
    multi_layer_loss_different = loss_fn_cd([different_kernels, layer2_different])
    
    print(f"Multi-layer similar kernels loss (Margin 0.1): {multi_layer_loss_similar.item():.6f}")
    print(f"Multi-layer different kernels loss (Margin 0.1): {multi_layer_loss_different.item():.6f}")
    
    # Compare normal mode vs channel diversity mode
    print("\n" + "="*50)
    print("COMPARING NORMAL MODE vs CHANNEL DIVERSITY MODE")
    print("="*50)
    
    # Convert channel diversity kernels to normal mode (flatten input channels)
    normal_mode_similar = similar_kernels.view(-1, kernel_size, kernel_size)
    normal_mode_different = different_kernels.view(-1, kernel_size, kernel_size)
    
    print(f"Normal mode shape: {normal_mode_similar.shape}")
    print(f"Channel diversity shape: {similar_kernels.shape}")
    
    loss_normal_similar = loss_fn_cd([normal_mode_similar])
    loss_normal_different = loss_fn_cd([normal_mode_different])
    loss_cd_similar = loss_fn_cd([similar_kernels])
    loss_cd_different = loss_fn_cd([different_kernels])
    
    print(f"\nSimilar kernels:")
    print(f"  Normal mode loss: {loss_normal_similar.item():.6f}")
    print(f"  Channel diversity loss: {loss_cd_similar.item():.6f}")
    print(f"  Difference: {abs(loss_normal_similar.item() - loss_cd_similar.item()):.6f}")
    
    print(f"\nDifferent kernels:")
    print(f"  Normal mode loss: {loss_normal_different.item():.6f}")
    print(f"  Channel diversity loss: {loss_cd_different.item():.6f}")
    print(f"  Difference: {abs(loss_normal_different.item() - loss_cd_different.item()):.6f}")
    
    print("\nNote: Channel diversity mode should generally produce different loss values")
    print("because it only compares kernels within the same input channel,")
    print("while normal mode compares all kernels with each other.")
