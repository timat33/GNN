import numpy as np


# def squared_exponential_kernel(x1, x2, h=1.0): # From the lecturer, causing issues
#     """Squared Exponential Kernel (RBF Kernel)"""
#     return np.exp(-np.sum((x1 - x2) ** 2) / h)


# def inverse_multiquadratic_kernel(x1, x2, h=1.0):
#     """Inverse Multiquadratic Kernel"""
#     return 1 / (np.sum((x1 - x2) ** 2) / h + 1)

def squared_exponential_kernel(x1, x2, h=1.0): # From online sources
    """Squared Exponential Kernel (RBF Kernel)"""
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * h ** 2))


def inverse_multiquadratic_kernel(x1, x2, h=1.0):
    """Inverse Multiquadratic Kernel"""
    return 1 / np.sqrt(np.sum((x1 - x2) ** 2) + h)


def calculate_mmd(X_true, X_pred, bandwidths=[1.0]):
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two distributions
    using both squared exponential and inverse multiquadratic kernels.

    Args:
        X_true (np.ndarray): Samples from distribution P, shape (N, D)
        X_pred (np.ndarray): Samples from distribution Q, shape (M, D)
        bandwidths (list): List of bandwidths

    Returns:
        dict: MMD values for each kernel type
    """

    # Define available kernels
    kernels = {
        "squared_exponential": squared_exponential_kernel,
        "inverse_multiquadratic": inverse_multiquadratic_kernel
    }

    # Dictionary to store MMD results for each kernel
    mmd_squared_results = {}

    # Loop over each kernel
    for kernel_name, kernel_fn in kernels.items():
        # Initialize components
        N = X_true.shape[0]
        M = X_pred.shape[0]
        xx_sum, yy_sum, xy_sum = 0.0, 0.0, 0.0

        # Calculate intra-distribution kernel sums for each bandwidth
        for h in bandwidths:
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    xx_sum += kernel_fn(X_true[i], X_true[j], h)
            for i in range(M):
                for j in range(M):
                    if i == j: # skip case with equal indices
                        continue
                    yy_sum += kernel_fn(X_pred[i], X_pred[j], h)
            for i in range(N):
                for j in range(M):
                    xy_sum += kernel_fn(X_true[i], X_pred[j], h)

            # Normalize sums by sample size
            xx_sum /= (N * (N-1))
            yy_sum /= (M * (M-1))
            xy_sum /= (N * M)

        # Calculate MMD for the current kernel
        mmd_squared_value = xx_sum + yy_sum - 2 * xy_sum
        mmd_squared_results[kernel_name] = mmd_squared_value

    # Find max squared value and take square root to return
    mmd = max(mmd_squared_results.values()) ** 0.5

    return mmd