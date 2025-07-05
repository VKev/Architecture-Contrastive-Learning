# contrastive_kernel_loss.py  •  chunk-aware, GPU-friendly implementation
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.metrics import ssim as kornia_ssim

# -----------------------------------------------------------------------------#
# helper: quick SSIM between two (H,W) kernels – kept for your own tests
# -----------------------------------------------------------------------------#
def _ssim_pair_gpu(k1: torch.Tensor, k2: torch.Tensor, win_size: int | None = None) -> float:
    k1, k2 = k1.float(), k2.float()
    h, w   = k1.shape
    if win_size is None:
        win_size = min(h, w) if min(h, w) & 1 else min(h, w) - 1
        win_size = min(win_size, 11)
    win_size = max(3, win_size | 1)
    rng = float((torch.max(k1.max(), k2.max()) - torch.min(k1.min(), k2.min())).item() or 1.0)
    return float(kornia_ssim(k1[None, None], k2[None, None], window_size=win_size, max_val=rng).mean())


# -----------------------------------------------------------------------------#
# MAIN LOSS (chunked)
# -----------------------------------------------------------------------------#
class ContrastiveKernelLoss(nn.Module):
    r"""
    SSIM-hinge loss for convolution kernels.

        hinge_ij = max(0, SSIM(k_i , k_j) – margin)

    *Normal mode*   : W.shape = (N , h , w)   → compare all unordered pairs  
    *Channel mode*  : W.shape = (C_in , C_out , h , w)  
                      → pairs **within each input-channel** only

    If the number of pairs is too large, they are processed in GPU-friendly
    **chunks** (≈200 k pairs by default) to avoid OOM.
    """

    # cache triangular indices so we build them once per (N, device)
    _triu_cache: dict[tuple[int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _get_triu_indices(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (n, device.type, device.index or 0)
        if key not in ContrastiveKernelLoss._triu_cache:
            ContrastiveKernelLoss._triu_cache[key] = torch.triu_indices(
                n, n, 1, device=device, dtype=torch.long
            )
        return ContrastiveKernelLoss._triu_cache[key]

    @staticmethod
    def _pairwise_ssim(k: torch.Tensor, i: torch.Tensor, j: torch.Tensor, win: int) -> torch.Tensor:
        """SSIM for selected pairs of kernels (all inputs on same device)."""
        rng = float((k.max() - k.min()).item() or 1.0)
        ssim = kornia_ssim(k[i, None], k[j, None], window_size=win, max_val=rng)
        return ssim.mean(dim=tuple(range(1, ssim.ndim)))  # → (pairs,)

    # ------------------------------------------------------------------ #
    def __init__(self,
                 margin: float = 0.9,
                 win_size: int | None = None,
                 pair_batch: int = 200_000):
        """
        margin       – similarities above this are penalised (0<margin<1)
        win_size     – odd SSIM window ≤11; computed automatically if None
        pair_batch   – max #pairs fed to a single Kornia call (controls memory)
        """
        super().__init__()
        assert 0. < margin < 1.
        self.margin      = margin
        self.win_size    = win_size
        self.pair_batch  = max(1, pair_batch)

    # ------------------------------------------------------------------ #
    def _layer_loss_pairs(self,
                          kernels: torch.Tensor,
                          i_idx: torch.Tensor,
                          j_idx: torch.Tensor,
                          ch_id: torch.Tensor | None,
                          win: int,
                          n_channels: int | None) -> torch.Tensor:
        """Compute hinge loss over (possibly chunked) pair list."""
        device = kernels.device
        if ch_id is None:                      # ----- normal mode -----
            hinge_sum = 0.0
            count_pos = 0
            for start in range(0, i_idx.numel(), self.pair_batch):
                s = slice(start, start + self.pair_batch)
                hinge = F.relu(self._pairwise_ssim(kernels, i_idx[s], j_idx[s], win) - self.margin)
                hinge_sum += hinge[hinge > 0].sum()
                count_pos += (hinge > 0).sum()
            return hinge_sum / max(count_pos, 1)

        else:                                  # --- channel mode ---
            sums   = torch.zeros(n_channels, device=device)
            counts = torch.zeros_like(sums)

            for start in range(0, i_idx.numel(), self.pair_batch):
                s      = slice(start, start + self.pair_batch)
                hinge  = F.relu(self._pairwise_ssim(kernels, i_idx[s], j_idx[s], win) - self.margin)
                pos    = hinge > 0
                if pos.any():
                    sums.index_add_(0, ch_id[s][pos], hinge[pos])
                    counts.index_add_(0, ch_id[s][pos], torch.ones_like(hinge[pos]))
            return (sums / counts.clamp(min=1)).mean()

    # ------------------------------------------------------------------ #
    def forward(self, kernels_list: List[torch.Tensor]) -> torch.Tensor:
        device, total, n_layers = kernels_list[0].device, 0.0, 0

        for W in kernels_list:
            if W.ndim == 4:                              # CHANNEL-DIVERSITY
                C_in, C_out, h, w = W.shape
                assert h == w
                win = self.win_size or (h if h & 1 else h - 1)
                win = min(max(win, 3), 11)

                flat   = W.reshape(C_in * C_out, h, w)
                base_i, base_j = self._get_triu_indices(C_out, device)
                P_per_ch = base_i.numel()
                offs   = torch.arange(C_in, device=device) * C_out
                i_idx  = (base_i[:, None] + offs).flatten()
                j_idx  = (base_j[:, None] + offs).flatten()
                ch_id  = torch.repeat_interleave(torch.arange(C_in, device=device), P_per_ch)

                layer_loss = self._layer_loss_pairs(flat, i_idx, j_idx, ch_id, win, C_in)

            else:                                       # NORMAL MODE
                N, h, w = W.shape
                assert h == w
                win = self.win_size or (h if h & 1 else h - 1)
                win = min(max(win, 3), 11)

                i_idx, j_idx = self._get_triu_indices(N, device)
                layer_loss   = self._layer_loss_pairs(W, i_idx, j_idx, None, win, None)

            total += layer_loss
            n_layers += 1

        return total / max(n_layers, 1)


# -----------------------------------------------------------------------------#
# Quick smoke-test (same as before, now OOM-safe)
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # tiny demo kernels
    small = torch.randn(64, 3, 3, device=device)
    lossfn = ContrastiveKernelLoss(margin=0.1).to(device)
    print("small demo loss:", lossfn([small]).item())

    # 256×256×7×7 channel-diversity stress test (previously OOM on 4 GB)
    Cin, Cout, k = 256, 256, 7
    big = torch.randn(Cin, Cout, k, k, device=device) * 0.05
    torch.cuda.empty_cache() if device.type == "cuda" else None
    t0 = time.time()
    val = lossfn([big])
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    print(f"big loss: {val.item():.6f} – {1000*(time.time()-t0):.1f} ms "
          f"(chunk size = {lossfn.pair_batch})")

    # -----------------------------------------------------------------------------#
    # COMPREHENSIVE SIMILARITY/DIFFERENCE TESTS
    # -----------------------------------------------------------------------------#
    print("\n" + "="*60)
    print("COMPREHENSIVE SIMILARITY/DIFFERENCE TESTS")
    print("="*60)

    # Test parameters
    margin_test = 0.5
    lossfn_test = ContrastiveKernelLoss(margin=margin_test).to(device)
    
    # Small tests (8 kernels of 3x3)
    print(f"\n--- SMALL TESTS (8 kernels, 3x3) ---")
    
    # Small similar - all kernels are very similar
    print("Small Similar:")
    base_kernel = torch.randn(1, 3, 3, device=device)
    small_similar = base_kernel + torch.randn(8, 3, 3, device=device) * 0.01  # Very small noise
    loss_small_similar = lossfn_test([small_similar])
    print(f"  Loss: {loss_small_similar.item():.6f} (should be high, > {margin_test})")
    
    # Small different - all kernels are very different
    print("Small Different:")
    small_different = torch.randn(8, 3, 3, device=device) * 2.0  # Large variation
    loss_small_different = lossfn_test([small_different])
    print(f"  Loss: {loss_small_different.item():.6f} (should be low, < {margin_test})")
    
    # Big tests (64 kernels of 7x7)
    print(f"\n--- BIG TESTS (64 kernels, 7x7) ---")
    
    # Big similar - all kernels are very similar
    print("Big Similar:")
    base_kernel_big = torch.randn(1, 7, 7, device=device)
    big_similar = base_kernel_big + torch.randn(64, 7, 7, device=device) * 0.01  # Very small noise
    torch.cuda.empty_cache() if device.type == "cuda" else None
    t0 = time.time()
    loss_big_similar = lossfn_test([big_similar])
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    print(f"  Loss: {loss_big_similar.item():.6f} (should be high, > {margin_test}) – {1000*(time.time()-t0):.1f} ms")
    
    # Big different - all kernels are very different
    print("Big Different:")
    big_different = torch.randn(64, 7, 7, device=device) * 2.0  # Large variation
    torch.cuda.empty_cache() if device.type == "cuda" else None
    t0 = time.time()
    loss_big_different = lossfn_test([big_different])
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    print(f"  Loss: {loss_big_different.item():.6f} (should be low, < {margin_test}) – {1000*(time.time()-t0):.1f} ms")
    
    # -----------------------------------------------------------------------------#
    # SPECIAL CHANNEL DIVERSITY TEST
    # -----------------------------------------------------------------------------#
    print(f"\n--- SPECIAL CHANNEL DIVERSITY TEST ---")
    print("Creating 256×256×7×7 tensor where:")
    print("- Within each input channel: all 256 kernels are similar")
    print("- Across input channels: kernels are different")
    
    # Create the special tensor
    Cin, Cout, k = 256, 256, 7
    
    # Create base patterns for each input channel (different from each other)
    base_patterns = torch.randn(Cin, 1, k, k, device=device) * 2.0  # Different base for each input channel
    
    # For each input channel, create similar kernels with small variations
    special_tensor = torch.zeros(Cin, Cout, k, k, device=device)
    for i in range(Cin):
        # All kernels in this input channel are similar to the base pattern
        noise = torch.randn(Cout, k, k, device=device) * 0.02  # Very small noise
        special_tensor[i] = base_patterns[i] + noise
    
    print(f"Tensor shape: {special_tensor.shape}")
    print(f"Base patterns shape: {base_patterns.shape}")
    
    # Test with ContrastiveKernelLoss (channel diversity mode)
    print("Testing with ContrastiveKernelLoss (channel diversity mode):")
    torch.cuda.empty_cache() if device.type == "cuda" else None
    t0 = time.time()
    loss_special = lossfn_test([special_tensor])
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    print(f"  Loss: {loss_special.item():.6f} – {1000*(time.time()-t0):.1f} ms")
    print(f"  Expected: High loss (kernels within each channel are similar)")
    
    # Verify our construction by checking similarity within and across channels
    print("\nVerifying construction:")
    
    # Check similarity within first few input channels
    for ch in range(min(3, Cin)):
        # Take first 5 kernels from this channel
        kernels_in_ch = special_tensor[ch][:5]  # Shape: (5, 7, 7)
        ssim_within = []
        for i in range(5):
            for j in range(i+1, 5):
                ssim_val = _ssim_pair_gpu(kernels_in_ch[i], kernels_in_ch[j])
                ssim_within.append(ssim_val)
        avg_ssim_within = sum(ssim_within) / len(ssim_within)
        print(f"  Channel {ch}: Avg SSIM within channel = {avg_ssim_within:.4f} (should be high)")
    
    # Check similarity across different input channels
    ssim_across = []
    for i in range(min(3, Cin)):
        for j in range(i+1, min(3, Cin)):
            # Compare first kernel from channel i with first kernel from channel j
            ssim_val = _ssim_pair_gpu(special_tensor[i][0], special_tensor[j][0])
            ssim_across.append(ssim_val)
    if ssim_across:
        avg_ssim_across = sum(ssim_across) / len(ssim_across)
        print(f"  Across channels: Avg SSIM = {avg_ssim_across:.4f} (should be low)")
    
    # Compare with random tensor of same size
    print("\nComparison with random tensor:")
    random_tensor = torch.randn(Cin, Cout, k, k, device=device) * 0.5
    torch.cuda.empty_cache() if device.type == "cuda" else None
    t0 = time.time()
    loss_random = lossfn_test([random_tensor])
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    print(f"  Random tensor loss: {loss_random.item():.6f} – {1000*(time.time()-t0):.1f} ms")
    print(f"  Special tensor loss: {loss_special.item():.6f}")
    print(f"  Ratio (special/random): {loss_special.item()/loss_random.item():.2f}")
    
    # Summary
    print(f"\n--- SUMMARY ---")
    print(f"Small similar loss:  {loss_small_similar.item():.6f}")
    print(f"Small different loss: {loss_small_different.item():.6f}")
    print(f"Big similar loss:    {loss_big_similar.item():.6f}")
    print(f"Big different loss:  {loss_big_different.item():.6f}")
    print(f"Special channel diversity loss: {loss_special.item():.6f}")
    print(f"Random tensor loss:  {loss_random.item():.6f}")
    print(f"Margin used: {margin_test}")
