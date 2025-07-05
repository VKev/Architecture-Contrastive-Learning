import random
import torch.nn as nn
import torch

def get_kernel_weight_matrix(weight, ignore_sizes=[1, 3], channel_diversity=False):

    if weight.shape[1] != 1:
        if channel_diversity:
            # Permute to group by input channels: (out_ch, in_ch, h, w) -> (in_ch, out_ch, h, w)
            kernel_matrix = weight.permute(1, 0, 2, 3)
        else:
            kernel_matrix = weight.mean(dim=1) 
    else:
        kernel_matrix = weight.squeeze(1) 

    k_size = kernel_matrix.shape[-1]
    if k_size in ignore_sizes:
        return None
    else:
        return kernel_matrix


def get_kernel_list(model, channel_diversity=False):
    """Extract kernel list from a model."""
    kernel_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            filtered_kernels = get_kernel_weight_matrix(
                module.weight, ignore_sizes=[1], channel_diversity=channel_diversity
            )
            if filtered_kernels is not None:
                kernel_list.append(filtered_kernels)
    return kernel_list

def select_random_kernels(kernel_list, k=12):
    """Select random kernels from each layer."""
    selected = []
    for kernels in kernel_list:
        if len(kernels.shape) == 4:  # Channel diversity mode: (in_ch, out_ch, h, w)
            in_channels, out_channels = kernels.shape[0], kernels.shape[1]
            if k <= 1.0:  # Percentage mode
                k_actual = max(1, int(out_channels * k))  # At least 1 kernel per input channel
            else:  # Integer mode
                k_actual = min(int(k), out_channels)
            
            # Select k kernels from each input channel
            selected_per_channel = []
            for i in range(in_channels):
                selected_indices = random.sample(range(out_channels), k_actual)
                selected_per_channel.append(kernels[i][selected_indices])
            
            # Stack to get (in_ch, k_actual, h, w)
            selected.append(torch.stack(selected_per_channel, dim=0))
        else:  # Normal mode: (out_ch, h, w)
            N = kernels.shape[0]
            if k <= 1.0:  # Percentage mode
                k_actual = max(1, int(N * k))  # At least 1 kernel
            else:  # Integer mode
                k_actual = min(int(k), N)
            selected_indices = random.sample(range(N), k_actual)
            selected.append(kernels[selected_indices])
    return selected

def select_fixed_kernels(kernel_list, k=12, seed=42):
    """Select fixed kernels from each layer using a consistent seed."""
    selected = []
    torch.manual_seed(seed)  # Set seed for reproducible selection
    
    for kernels in kernel_list:
        if len(kernels.shape) == 4:  # Channel diversity mode: (in_ch, out_ch, h, w)
            in_channels, out_channels = kernels.shape[0], kernels.shape[1]
            if k <= 1.0:  # Percentage mode
                k_actual = max(1, int(out_channels * k))  # At least 1 kernel per input channel
            else:  # Integer mode
                k_actual = min(int(k), out_channels)
            
            # Select k kernels from each input channel using consistent seed
            selected_per_channel = []
            for i in range(in_channels):
                selected_indices = torch.randperm(out_channels)[:k_actual].tolist()
                selected_per_channel.append(kernels[i][selected_indices])
            
            # Stack to get (in_ch, k_actual, h, w)
            selected.append(torch.stack(selected_per_channel, dim=0))
        else:  # Normal mode: (out_ch, h, w)
            N = kernels.shape[0]
            if k <= 1.0:  # Percentage mode
                k_actual = max(1, int(N * k))  # At least 1 kernel
            else:  # Integer mode
                k_actual = min(int(k), N)
            
            # Use torch.randperm for consistent selection
            selected_indices = torch.randperm(N)[:k_actual].tolist()
            selected.append(kernels[selected_indices])
    
    return selected

def test_kernel_functions():
    import torch
    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from model.resnet_cifar import resnet20
    
    print("="*50)
    print("Testing kernel functions with ResNet20")
    print("="*50)
    
    model = resnet20()
    sample_input = torch.randn(1, 3, 32, 32)
    
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Sample input shape: {sample_input.shape}")
    
    print("\n--- Forward pass test ---")
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    
    print("\n--- Kernel extraction test (normal mode) ---")
    kernel_list = get_kernel_list(model)
    conv_layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer_count += 1
            print(f"Conv layer {conv_layer_count}: {name}")
            print(f"  Weight shape: {module.weight.shape}")
            
            filtered_kernels = get_kernel_weight_matrix(
                module.weight, ignore_sizes=[1]
            )
            
            if filtered_kernels is not None:
                print(f"  Filtered kernels shape: {filtered_kernels.shape}")
            else:
                print(f"  Kernels ignored (size {module.weight.shape[-1]})")
            print()
    
    print(f"Total conv layers: {conv_layer_count}")
    print(f"Layers with extracted kernels: {len(kernel_list)}")
    
    print("\n--- Kernel extraction test (channel diversity mode) ---")
    kernel_list_cd = get_kernel_list(model, channel_diversity=True)
    conv_layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer_count += 1
            print(f"Conv layer {conv_layer_count}: {name}")
            print(f"  Weight shape: {module.weight.shape}")
            
            filtered_kernels = get_kernel_weight_matrix(
                module.weight, ignore_sizes=[1], channel_diversity=True
            )
            
            if filtered_kernels is not None:
                print(f"  Filtered kernels shape (channel diversity): {filtered_kernels.shape}")
            else:
                print(f"  Kernels ignored (size {module.weight.shape[-1]})")
            print()
    
    print(f"Layers with extracted kernels (channel diversity): {len(kernel_list_cd)}")
    
    if not kernel_list:
        print("No kernels extracted! Cannot test selection functions.")
        return
    
    print("\n--- Random kernel selection test ---")
    k_values = [0.1, 0.25, 0.5, 8, 16, 32]
    
    for k in k_values:
        print(f"\nTesting random selection with k={k}:")
        try:
            selected_kernels = select_random_kernels(kernel_list, k)
            total_original = sum(kernels.shape[0] for kernels in kernel_list)
            total_selected = sum(kernels.shape[0] for kernels in selected_kernels)
            
            for i, (orig, sel) in enumerate(zip(kernel_list, selected_kernels)):
                N = orig.shape[0]
                selected_count = sel.shape[0]
                if k < 1.0:
                    expected = max(1, int(N * k))
                    print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {selected_count}/{N}, expected ~{expected} for {k*100:.0f}%)")
                else:
                    expected = min(int(k), N)
                    print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {selected_count}/{N}, expected {expected})")
            
            print(f"  Total kernels: {total_original} -> {total_selected}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test fixed kernel selection
    print("\n--- Fixed kernel selection test ---")
    for k in k_values:
        print(f"\nTesting fixed selection with k={k}:")
        try:
            selected_kernels_1 = select_fixed_kernels(kernel_list, k, seed=42)
            selected_kernels_2 = select_fixed_kernels(kernel_list, k, seed=42)
            
            identical = True
            for sel1, sel2 in zip(selected_kernels_1, selected_kernels_2):
                if not torch.equal(sel1, sel2):
                    identical = False
                    break
            
            total_original = sum(kernels.shape[0] for kernels in kernel_list)
            total_selected = sum(kernels.shape[0] for kernels in selected_kernels_1)
            
            for i, (orig, sel) in enumerate(zip(kernel_list, selected_kernels_1)):
                N = orig.shape[0]
                selected_count = sel.shape[0]
                if k < 1.0:
                    expected = max(1, int(N * k))
                    print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {selected_count}/{N}, expected ~{expected} for {k*100:.0f}%)")
                else:
                    expected = min(int(k), N)
                    print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {selected_count}/{N}, expected {expected})")
            
            print(f"  Total kernels: {total_original} -> {total_selected}")
            print(f"  Selection consistency: {'✓ Identical' if identical else '✗ Different'}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n--- Edge case tests ---")
    print("Testing with k=0.1 (10%):")
    try:
        selected = select_random_kernels(kernel_list, 0.1)
        for i, (orig, sel) in enumerate(zip(kernel_list, selected)):
            print(f"  Layer {i+1}: {orig.shape[0]} -> {sel.shape[0]} (min 1 kernel enforced)")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTesting with k=1000 (very large):")
    try:
        selected = select_random_kernels(kernel_list, 1000)
        for i, (orig, sel) in enumerate(zip(kernel_list, selected)):
            print(f"  Layer {i+1}: {orig.shape[0]} -> {sel.shape[0]} (capped at available)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test channel diversity mode
    if kernel_list_cd:
        print("\n" + "="*60)
        print("TESTING CHANNEL DIVERSITY MODE")
        print("="*60)
        
        print("\n--- Channel diversity random selection test ---")
        k_values = [0.1, 0.25, 0.5, 4, 8, 16]  # Mix of percentage and integer values
        
        for k in k_values:
            print(f"\nTesting channel diversity random selection with k={k}:")
            try:
                selected_kernels = select_random_kernels(kernel_list_cd, k)
                total_original = sum(kernels.shape[0] * kernels.shape[1] for kernels in kernel_list_cd)
                total_selected = sum(kernels.shape[0] * kernels.shape[1] for kernels in selected_kernels)
                
                for i, (orig, sel) in enumerate(zip(kernel_list_cd, selected_kernels)):
                    in_ch, out_ch = orig.shape[0], orig.shape[1]
                    sel_in_ch, sel_out_ch = sel.shape[0], sel.shape[1]
                    if k <= 1.0:
                        expected_per_ch = max(1, int(out_ch * k))
                        expected_total = in_ch * expected_per_ch
                        print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {sel_in_ch}×{sel_out_ch}={sel_in_ch*sel_out_ch}, expected {in_ch}×{expected_per_ch}={expected_total})")
                    else:
                        expected_per_ch = min(int(k), out_ch)
                        expected_total = in_ch * expected_per_ch
                        print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {sel_in_ch}×{sel_out_ch}={sel_in_ch*sel_out_ch}, expected {in_ch}×{expected_per_ch}={expected_total})")
                
                print(f"  Total kernels: {total_original} -> {total_selected}")
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n--- Channel diversity fixed selection test ---")
        for k in k_values:
            print(f"\nTesting channel diversity fixed selection with k={k}:")
            try:
                # Test twice to ensure consistency
                selected_kernels_1 = select_fixed_kernels(kernel_list_cd, k, seed=42)
                selected_kernels_2 = select_fixed_kernels(kernel_list_cd, k, seed=42)
                
                # Check if selections are identical
                identical = True
                for sel1, sel2 in zip(selected_kernels_1, selected_kernels_2):
                    if not torch.equal(sel1, sel2):
                        identical = False
                        break
                
                total_original = sum(kernels.shape[0] * kernels.shape[1] for kernels in kernel_list_cd)
                total_selected = sum(kernels.shape[0] * kernels.shape[1] for kernels in selected_kernels_1)
                
                for i, (orig, sel) in enumerate(zip(kernel_list_cd, selected_kernels_1)):
                    in_ch, out_ch = orig.shape[0], orig.shape[1]
                    sel_in_ch, sel_out_ch = sel.shape[0], sel.shape[1]
                    if k < 1.0:
                        expected_per_ch = max(1, int(out_ch * k))
                        expected_total = in_ch * expected_per_ch
                        print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {sel_in_ch}×{sel_out_ch}={sel_in_ch*sel_out_ch}, expected {in_ch}×{expected_per_ch}={expected_total})")
                    else:
                        expected_per_ch = min(int(k), out_ch)
                        expected_total = in_ch * expected_per_ch
                        print(f"  Layer {i+1}: {orig.shape} -> {sel.shape} (selected {sel_in_ch}×{sel_out_ch}={sel_in_ch*sel_out_ch}, expected {in_ch}×{expected_per_ch}={expected_total})")
                
                print(f"  Total kernels: {total_original} -> {total_selected}")
                print(f"  Selection consistency: {'✓ Identical' if identical else '✗ Different'}")
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n--- Channel diversity edge case tests ---")
        print("Testing with k=0.1 (10%):")
        try:
            selected = select_random_kernels(kernel_list_cd, 0.1)
            for i, (orig, sel) in enumerate(zip(kernel_list_cd, selected)):
                in_ch, out_ch = orig.shape[0], orig.shape[1]
                sel_in_ch, sel_out_ch = sel.shape[0], sel.shape[1]
                print(f"  Layer {i+1}: {in_ch}×{out_ch} -> {sel_in_ch}×{sel_out_ch} (min 1 kernel per channel enforced)")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\nTesting with k=1000 (very large):")
        try:
            selected = select_random_kernels(kernel_list_cd, 1000)
            for i, (orig, sel) in enumerate(zip(kernel_list_cd, selected)):
                in_ch, out_ch = orig.shape[0], orig.shape[1]
                sel_in_ch, sel_out_ch = sel.shape[0], sel.shape[1]
                print(f"  Layer {i+1}: {in_ch}×{out_ch} -> {sel_in_ch}×{sel_out_ch} (capped at available per channel)")
        except Exception as e:
            print(f"  Error: {e}")


def test_integration_with_loss():
    """Test integration between selective kernel functions and contrastive loss."""
    import torch
    import sys
    import os
    
    # Add parent directory to path to import from model and util
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from model.resnet_cifar import resnet20
        from util.loss import ContrastiveKernelLoss
    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("Skipping integration test...")
        return
    
    print("\n" + "="*70)
    print("INTEGRATION TEST: SELECTIVE KERNELS + CONTRASTIVE LOSS")
    print("="*70)
    
    # Create model
    model = resnet20()
    
    # Test both modes
    modes = [
        ("Normal Mode", False),
        ("Channel Diversity Mode", True)
    ]
    
    for mode_name, channel_diversity in modes:
        print(f"\n--- {mode_name} ---")
        
        # Extract kernels
        kernel_list = get_kernel_list(model, channel_diversity=channel_diversity)
        
        if not kernel_list:
            print(f"No kernels extracted in {mode_name}")
            continue
        
        print(f"Extracted {len(kernel_list)} layers")
        
        # Test different k values
        k_values = [0.25, 8]
        
        for k in k_values:
            print(f"\nTesting with k={k}:")
            
            # Select kernels
            selected_kernels = select_random_kernels(kernel_list, k)
            
            # Print shapes
            total_orig = 0
            total_selected = 0
            
            for i, (orig, sel) in enumerate(zip(kernel_list, selected_kernels)):
                if channel_diversity:
                    orig_count = orig.shape[0] * orig.shape[1]
                    sel_count = sel.shape[0] * sel.shape[1]
                    print(f"  Layer {i+1}: {orig.shape} -> {sel.shape}")
                else:
                    orig_count = orig.shape[0]
                    sel_count = sel.shape[0]
                    print(f"  Layer {i+1}: {orig.shape} -> {sel.shape}")
                
                total_orig += orig_count
                total_selected += sel_count
            
            print(f"  Total kernels: {total_orig} -> {total_selected}")
            
            # Test with ContrastiveKernelLoss
            try:
                loss_fn = ContrastiveKernelLoss(margin=0.5, win_size=None)
                loss_value = loss_fn(selected_kernels)
                print(f"  Contrastive loss: {loss_value.item():.6f}")
            except Exception as e:
                print(f"  Loss calculation error: {e}")
        
        # Test fixed selection for consistency
        print(f"\nTesting fixed selection consistency:")
        try:
            selected_1 = select_fixed_kernels(kernel_list, k=8, seed=42)
            selected_2 = select_fixed_kernels(kernel_list, k=8, seed=42)
            
            # Check consistency
            identical = True
            for sel1, sel2 in zip(selected_1, selected_2):
                if not torch.equal(sel1, sel2):
                    identical = False
                    break
            
            print(f"  Fixed selection consistency: {'✓ Identical' if identical else '✗ Different'}")
            
            # Test loss consistency
            loss_1 = ContrastiveKernelLoss(margin=0.5)(selected_1)
            loss_2 = ContrastiveKernelLoss(margin=0.5)(selected_2)
            loss_identical = abs(loss_1.item() - loss_2.item()) < 1e-6
            print(f"  Loss consistency: {'✓ Identical' if loss_identical else '✗ Different'} ({loss_1.item():.6f} vs {loss_2.item():.6f})")
            
        except Exception as e:
            print(f"  Consistency test error: {e}")


if __name__ == "__main__":
    test_kernel_functions()
    test_integration_with_loss()