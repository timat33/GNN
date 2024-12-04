from scipy.stats import multivariate_normal
import pandas as pd
import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from train_model_conditional import RealNVP, get_standardised_moons, get_standardised_gmm
# hparam analysis functions
def get_path_from_params(n_train, hidden_size, blocks, lr, dataset = 'moons', conditional = False):
    return f'models/{dataset}{"_conditional" if conditional else ""}/{dataset}_INN_ntrain{int(n_train)}_hiddensize{int(hidden_size)}_blocks{int(blocks)}_lr{str(lr).replace(".",",")}.pt'

def find_best_model_conditional(min_losses_df, dataset = 'moons', conditional = False):
    print(min_losses_df)

    best_params = min_losses_df.loc[min_losses_df['min_val_loss'].idxmin()]

    # Return path of best models
    best_path = get_path_from_params(best_params['n_train'], best_params['hidden_size'], best_params['blocks'], best_params['lr'],
                                     dataset, conditional)
    

    return best_path, best_params

# MMD helper functions
def squared_exponential_kernel(a, b, h=1):
    a_expanded = a.unsqueeze(1)  # Shape: (N, 1, D)
    b_expanded = b.unsqueeze(0)  # Shape: (1, M, D)
    diff = a_expanded - b_expanded  # Shape: (N, M, D)
    result = torch.exp(-torch.sum(diff ** 2, dim=2) / h)  # Shape: (N, M)
    return result

def inverse_multiquad_kernel(a, b, h=1):
    a_expanded = a.unsqueeze(1)  # Shape: (N, 1, D)
    b_expanded = b.unsqueeze(0)  # Shape: (1, M, D)
    diff = a_expanded - b_expanded  # Shape: (N, M, D)
    result = torch.pow(torch.sum(diff ** 2, dim=2) / h + 1, -1)  # Shape: (N, M)
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
def check_normality(z:torch.Tensor, labels = None, bandwidth_start = 0.5, num_bandwidths = 5, seed = 11121):
    if seed:
        torch.manual_seed(seed)
 
    # ToDo make torch native
    z_norm = torch.randn(len(z), 2)
    
    mmd2 = MMD2(z, z_norm, bandwidth_start, num_bandwidths)

    print(f'Squared MMD between codes and true normal sample: {mmd2}')


    # Put contours of normal plot here
    plt.figure(figsize=(8, 6))
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    Z = rv.pdf(pos)
    plt.contour(X, Y, Z, levels=10, cmap='Reds', alpha=0.5)
    if labels is not None:
        # Get unique labels and assign colors
        unique_labels = torch.unique(labels)
        colors = plt.cm.tab10(range(len(unique_labels)))
        
        # Plot each label class separately
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(z[mask, 0], z[mask, 1],
                       c=[colors[i]], 
                       marker='^',
                       label=f'Class {label.item()}',
                       alpha=0.6
                       )
    else:
        plt.scatter(z[:, 0], z[:, 1], 
            c='blue', marker='^')
    
    plt.title(f'Code distribution: MMD^2 = {mmd2}, \nvar = {torch.var(z)}, mean = {z.mean()}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_generation_quality(x_synth, x_test, quality, labels_synth=None, labels_test=None, bandwidth_start = 0.5, num_bandwidths = 5):
    mmd2 = MMD2(x_synth, x_test, bandwidth_start, num_bandwidths)

    # plot synthetic data
    if labels_synth is None:
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_test[:, 0], x_test[:, 1], 
                c='blue', marker='^', label='Synthetic Data', alpha = 0.6)
    else:
        # Get unique labels and assign colors
        unique_labels = torch.unique(labels_synth)
        colors = plt.cm.tab10(range(len(unique_labels)))
        
        # Plot each label class separately
        for i, label in enumerate(unique_labels):
            mask = labels_synth == label
            plt.scatter(x_synth[mask, 0], x_synth[mask, 1],
                       c=[colors[i]], 
                       marker='x',
                       label=f'Synthetic Class {int(label.item())}', 
                       alpha = 0.6
                       )
            
    # Plot true data
    if labels_test is None:
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_test[:, 0], x_test[:, 1], 
                c='blue', marker='^', label='Synthetic Data', alpha = 0.6)
    else:
        # Get unique labels and assign colors
        unique_labels = torch.unique(labels_test)
        colors = plt.cm.tab10(range(len(unique_labels)))
        
        # Plot each label class separately
        for i, label in enumerate(unique_labels):
            mask = labels_test == label
            plt.scatter(x_test[mask, 0], x_test[mask, 1],
                       c=[colors[i]], 
                       marker='^',
                       label=f'True Class {label.item()}',
                       alpha = 0.6
                )

    plt.title(f'Test vs Synthetic Points ({quality} quality: MMD^2 = {mmd2})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_conditional_analysis(best_path, best_params, dataset, noise = 0.1, seed = 11121, n_test = 100):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Get models
    input_size = 2
    best_inn = RealNVP(input_size, int(best_params['hidden_size']), int(best_params['blocks']), condition_size=1, device='cpu')
    checkpoint = torch.load(best_path, weights_only=False)
    best_inn.load_state_dict(checkpoint['model_state_dict'])


    # Get various datasets
    best_params
    if dataset == 'moons':
        x_test, labels_test = get_standardised_moons(int(best_params['n_train']), noise)
    elif dataset == 'gmm':
        x_test, labels_test = get_standardised_gmm(int(best_params['n_train']), noise)
    z_test = best_inn.get_codes(x_test, labels_test) # Codes corresponding to test data
    x_synth_good, labels_synth = best_inn.sample(n_test, conditions = [0,1], seed = seed) # good synthetic data

    # Run checks
    check_normality(z_test, labels_test)

    check_generation_quality(x_synth_good, x_test, quality = 'good', labels_synth = labels_synth, labels_test = labels_test)

if __name__ == '__main__':
    # Hyperparam analysis
    min_losses_df = pd.read_csv('min_losses_moons.csv')

    # Tabular analysis: Which is best model
    best_path, best_params = find_best_model_conditional(min_losses_df, dataset = 'moons', conditional = True)

    run_conditional_analysis(best_path, best_params, 'moons')

    # Rerun analysis for gmm

    # # Tabular analysis: Which is best model
    # best_path, best_params = find_best_model_conditional(min_losses_df, dataset = 'gmm', conditional = True)

    # run_conditional_analysis(best_path, best_params, 'gmm')
    