import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from train_model import RealNVP, get_standardised_moons, seed, noise, input_size, hidden_size, blocks, device, best_model_path

# MMD helper functions
def squared_exponential_kernel(a, b, h = 1):
    result = torch.empty((a.shape[0], b.shape[0]))

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            result[i, j] = torch.exp(-torch.sum(torch.pow(a[i]-b[j], 2))/h)
    return result

def inverse_multiquad_kernel(a, b, h = 1):
    result = torch.empty((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            result[i, j] = torch.pow((torch.sum(torch.pow(a[i]-b[j], 2))/h+1), -1)  
    return result

def MMD2(prediction_set, true_set, bandwidth_start, num_bandwidths):
    bandwidths = [bandwidth_start*2**i for i in range(num_bandwidths)]
    # Define kernel functions to use: Sum of squared exponential kernels or 
    # sum of inverse multiquadratic kernels.
    kernels = [squared_exponential_kernel, inverse_multiquad_kernel]

    # Convert to np array if needed
    if isinstance(prediction_set, list):
        prediction_set = np.array(prediction_set)
    if isinstance(true_set, list):
        true_set = np.array(true_set)

    # Find number of predictions and true samples
    N = prediction_set.shape[0]
    M = true_set.shape[0]

    # Calculate squared mean discrepancy per kernel and return maximum
    squared_mean_discrepancies = torch.zeros(len(kernels))
    for i, kernel_unbandwidthed in enumerate(kernels):
        kernel_squared_mean_disrepancy = 0 # will be the sum of all the bandwidthed kernels)\
        for bandwidth in bandwidths:
            kernel = lambda a, b: kernel_unbandwidthed(a, b, bandwidth)
            bandwidth_squared_mean_discrepancy = \
                kernel(true_set, true_set).sum() - kernel(true_set, true_set).diag().sum()/(N*(N-1)) + \
                kernel(prediction_set, prediction_set).sum()-kernel(prediction_set, prediction_set).diag().sum()/(M*(M-1)) - \
                2*kernel(prediction_set, true_set).sum()/(N*M)
            kernel_squared_mean_disrepancy += bandwidth_squared_mean_discrepancy
        
        squared_mean_discrepancies[i] = kernel_squared_mean_disrepancy
    
    mmd2 = squared_mean_discrepancies.max()

    return mmd2

# Analysis helper functions
def check_2d_reconstructions(x, x_reconstructed):
    # Check that reconstructions are approximately the same as the inputs
    eps = 1e-5
    are_close = torch.all_close(x, x_reconstructed, atol = eps)
    print(f'All reconstructions are within {eps} of the original points: {are_close}')

    # Graph reconstructions
    x_test = x.cpu().numpy()
    x_reconstructed = x_reconstructed.cpu().numpy()

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test[:, 0], x_test[:, 1], 
            c='blue', marker='o', label='Original')
    plt.scatter(x_reconstructed[:, 0], x_reconstructed[:, 1], 
            c='red', marker='x', label='Reconstructed')
    
    plt.title('Original vs Reconstructed Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_normality(z:torch.Tensor, bandwidth_start = 0.5, num_bandwidths = 5, seed = 11121):
    if seed:
        torch.manual_seed(seed)
 
    # ToDo make torch native
    z_norm = torch.randn(len(z))
    
    mmd2 = MMD2(z, z_norm, bandwidth_start, num_bandwidths)

    print(f'Squared MMD between codes and true normal sample: {mmd2}')

    # Put contours of normal plot here

    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], 
            c='blue', marker='o')
    
    plt.title(f'Code distribution: MMD^2 = {mmd2}, \nvar = {torch.var(z)}, mean = {z.mean()}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def check_generation_quality(x_synth, x_test, bandwidth_start = 0.5, num_bandwidths = 5):
    mmd2 = MMD2(x_synth, x_test, bandwidth_start, num_bandwidths)

    print(f'Squared MMD between codes and true normal sample: {mmd2}')

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test[:, 0], x_test[:, 1], 
            c='blue', marker='o', label='Test Data')
    plt.scatter(x_synth[:, 0], x_synth[:, 1], 
            c='red', marker='x', label='Synthetic Data')
    
    plt.title('Test vs Synthetic Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()



# New inference params
n_test = 100
test_batch_size = 32


# Get model
inn = RealNVP(input_size, hidden_size, blocks, device)
best_model_path = 'best_model_safe.pt' # TESTING
checkpoint = torch.load(best_model_path)
inn.load_state_dict(checkpoint['model_state_dict'])

# Analysis
# Get various datasets
x_test = get_standardised_moons(n_test, noise)

# Get codes and reconstructed inputs
z_test = inn.get_codes(x_test, test_batch_size)
x_reconstructed_test = inn.get_reconstructions(z_test, test_batch_size)

# Get synthetic data
x_synth = inn.sample(n_test, seed)

# Check reconstructions
check_2d_reconstructions(x_test, x_reconstructed_test)

# Check normality of codes
check_normality(z_test)

# Check synthetic quality:
print('Good quality:')
check_generation_quality(x_synth, x_test)

print('Bad quality:')
inn = RealNVP(input_size, hidden_size, blocks, device)
best_model_path = 'bad_model.pt' # TESTING
checkpoint = torch.load(best_model_path)
inn.load_state_dict(checkpoint['model_state_dict'])

x_synth = inn.sample(n_test, seed)
check_generation_quality(x_synth, x_test)

# Repeat experiments with GMM data:

