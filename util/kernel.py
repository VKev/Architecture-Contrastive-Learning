import random
import torch.nn as nn

def get_kernel_weight_matrix(weight, ignore_sizes=[1, 3]):

    if weight.shape[1] != 1:
        kernel_matrix = weight.mean(dim=1) 
    else:
        kernel_matrix = weight.squeeze(1) 

    k_size = kernel_matrix.shape[-1]
    if k_size in ignore_sizes:
        return None
    else:
        return kernel_matrix

def _get_kernel_list(self):
        kernel_list = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                filtered_kernels = get_kernel_weight_matrix(
                    module.weight, ignore_sizes=[1]
                )
                if filtered_kernels is not None:
                    kernel_list.append(filtered_kernels)
        return kernel_list

def _select_random_kernels(self, kernel_list, k=12):
    selected = []
    for kernels in kernel_list:
        N = kernels.shape[0]
        k = min(k, N)
        selected_indices = random.sample(range(N), k)
        selected.append(kernels[selected_indices])
    return selected


def test_kernel_functions():
    """Test kernel extraction functions with resnet20 model."""
    import torch
    import sys
    import os
    
    # Add parent directory to path to import from model
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from model.resnet_cifar import resnet20
    
    print("="*50)
    print("Testing kernel functions with ResNet20")
    print("="*50)
    
    # Create model and sample input
    model = resnet20()
    sample_input = torch.randn(1, 3, 32, 32)
    
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Sample input shape: {sample_input.shape}")
    
    # Test forward pass first
    print("\n--- Forward pass test ---")
    with torch.no_grad():
        output = model(sample_input)
    print(f"Output shape: {output.shape}")
    
    # Test kernel extraction
    print("\n--- Kernel extraction test ---")
    kernel_list = []
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
                kernel_list.append(filtered_kernels)
                print(f"  Filtered kernels shape: {filtered_kernels.shape}")
            else:
                print(f"  Kernels ignored (size {module.weight.shape[-1]})")
            print()
    
    print(f"Total conv layers: {conv_layer_count}")
    print(f"Layers with extracted kernels: {len(kernel_list)}")
    
    # Test random kernel selection
    if kernel_list:
        print("\n--- Random kernel selection test ---")
        k_values = [12, 32, 64]
        
        for k in k_values:
            print(f"\nTesting with k={k}:")
            selected_kernels = []
            
            for i, kernels in enumerate(kernel_list):
                N = kernels.shape[0]
                k_actual = min(k, N)
                selected_indices = random.sample(range(N), k_actual)
                selected = kernels[selected_indices]
                selected_kernels.append(selected)
                print(f"  Layer {i+1}: {kernels.shape} -> {selected.shape} (selected {k_actual}/{N})")
            
            total_selected = sum(k.shape[0] for k in selected_kernels)
            print(f"  Total kernels selected: {total_selected}")


if __name__ == "__main__":
    test_kernel_functions()