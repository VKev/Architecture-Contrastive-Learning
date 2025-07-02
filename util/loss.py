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
    """
    Ultra-fast GPU-accelerated Contrastive Kernel SSIM Loss using vectorized operations.
    Hinge loss on SSIM similarity:
        loss_ij = max(0, SSIM(k_i, k_j) - margin)
    (diagonal ignored, averaged over all pairs & layers)
    """
    def __init__(self, margin: float = 0.90, win_size: int | None = None):
        """
        margin : float in (0, 1) — similarity above this value is penalised.
        win_size : int | None   — odd window size for SSIM; set small (3/5)
                                   when kernels are tiny (3×3, 5×5).
        """
        super().__init__()
        self.margin = margin
        self.win_size = win_size

    def _vectorized_ssim(self, kernels, win_size):
        """
        Compute SSIM for all kernel pairs using vectorized GPU operations.
        """
        device = kernels.device
        n, h, w = kernels.shape
        
        # Ensure odd window size
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)
        
        # Calculate dynamic range for all kernels
        data_range = float((kernels.max() - kernels.min()).item())
        if data_range == 0:
            data_range = 1.0
        
        # Prepare all pairs for vectorized computation
        # Create indices for upper triangular pairs (i < j)
        i_indices, j_indices = torch.triu_indices(n, n, offset=1, device=device)
        
        # Get kernel pairs - shape: (num_pairs, 1, h, w)
        kernels_i = kernels[i_indices].unsqueeze(1)  # (num_pairs, 1, h, w)
        kernels_j = kernels[j_indices].unsqueeze(1)  # (num_pairs, 1, h, w)
        
        # Compute SSIM for all pairs at once
        ssim_values = kornia_ssim(kernels_i, kernels_j, window_size=win_size, max_val=data_range)
        
        # Handle different return shapes from kornia
        if ssim_values.ndim > 1:
            ssim_values = ssim_values.mean(dim=tuple(range(1, ssim_values.ndim)))
        
        # Create symmetric matrix
        ssim_mat = torch.zeros((n, n), device=device)
        ssim_mat[i_indices, j_indices] = ssim_values
        ssim_mat[j_indices, i_indices] = ssim_values  # Make symmetric
        
        return ssim_mat

    def forward(self, kernels_list):
        """
        kernels_list : list[Tensor] — each tensor shaped (n, d, d)
        """
        start_time = time.time()
        
        device = kernels_list[0].device
        total_loss, num_layers = 0.0, 0

        for layer_idx, kernels in enumerate(kernels_list):
            layer_start_time = time.time()
            n, d, d2 = kernels.shape
            assert d == d2, "Each kernel must be square"

            w = self.win_size
            if w is None:
                w = d if d % 2 == 1 else d - 1
                w = min(w, 11)

            # Vectorized SSIM computation
            ssim_start_time = time.time()
            ssim_mat = self._vectorized_ssim(kernels, w)
            ssim_time = time.time() - ssim_start_time

            # hinge: penalise high similarity
            loss_mat = F.relu(ssim_mat - self.margin)

            # average over off-diagonal pairs
            num_pairs = n * (n - 1)
            layer_loss = loss_mat.sum() / num_pairs
            total_loss += layer_loss
            num_layers += 1
            
            layer_time = time.time() - layer_start_time

        total_time = time.time() - start_time
        
        return total_loss / num_layers if num_layers else torch.tensor(0.0, device=device)


# A simple model with 3 convolutional layers.
class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # For simplicity, each conv has one input channel so that weight shape is (out_channels, 1, k, k)
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3, padding=1)  # kernels: (512, 1, 3, 3)
        self.conv2 = nn.Conv2d(1, 512, kernel_size=7, padding=3)  # kernels: (512, 1, 7, 7)
        self.conv3 = nn.Conv2d(1, 512, kernel_size=5, padding=2)   # kernels: (512, 1, 5, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

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
