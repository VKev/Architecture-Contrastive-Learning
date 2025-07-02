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