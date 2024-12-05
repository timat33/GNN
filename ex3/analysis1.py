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

from train_model import RealNVP, get_standardised_moons, get_standardised_gmm
# hparam analysis functions
def get_path_from_params(n_train, hidden_size, blocks, lr):
    return f'ex3/models/moons/moons_INN_ntrain{int(n_train)}_hiddensize{int(hidden_size)}_blocks{int(blocks)}_lr{str(lr).replace(".",",")}.pt'

def find_best_model_from_tabular(min_losses_df):
    print(min_losses_df)

    best_params = min_losses_df.loc[min_losses_df['min_val_loss'].idxmin()]
    worst_params = min_losses_df.loc[min_losses_df['min_val_loss'].idxmax()]

    # Create summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Best Loss', 'Worst Loss'],
        'Loss': [best_params['min_val_loss'], worst_params['min_val_loss']],
        'Hidden Size': [best_params['hidden_size'], worst_params['hidden_size']],
        'Blocks': [best_params['blocks'], worst_params['blocks']],
        'N Train': [best_params['n_train'], worst_params['n_train']],
        'Learning Rate': [best_params['lr'], worst_params['lr']]
    })
    print(summary)

    # Return path of best and worst models

    best_path = get_path_from_params(best_params['n_train'], best_params['hidden_size'], best_params['blocks'], best_params['lr'])
    bad_path = get_path_from_params(worst_params['n_train'], worst_params['hidden_size'], worst_params['blocks'], worst_params['lr'])

    return best_path, bad_path, best_params, worst_params
    
def plot_training_histories(path1, path2, model_names=None):
    """Compare training histories from two model checkpoints on single plot"""
    plt.figure(figsize=(10, 6))
    
    # Load histories
    hist1 = torch.load(path1)['history']
    hist2 = torch.load(path2)['history']
    
    # Plot settings
    names = model_names or ['Model 1', 'Model 2']
    
    # Plot first model
    epochs1 = range(1, len(hist1['train_loss']) + 1)
    plt.plot(epochs1, hist1['train_loss'], color='blue', linestyle='-', 
             label=f'{names[0]} Train Loss')
    plt.plot(epochs1, hist1['val_loss'], color='blue', linestyle='--',
             label=f'{names[0]} Val Loss')
    
    # Plot second model
    epochs2 = range(1, len(hist2['train_loss']) + 1)
    plt.plot(epochs2, hist2['train_loss'], color='red', linestyle='-',
             label=f'{names[1]} Train Loss')
    plt.plot(epochs2, hist2['val_loss'], color='red', linestyle='--',
             label=f'{names[1]} Val Loss')
    
    plt.ylim(0,5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all_parameter_changes(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    params = ['hidden_size', 'blocks', 'n_train', 'lr']
    
    for param, ax in zip(params, axes.flatten()):
        other_params = [col for col in df.columns if col not in [param, 'min_val_loss']]
        combinations = df.groupby(other_params)
        
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        for combo, group in combinations:
            group = group.sort_values(param)
            x1, x2 = group['min_val_loss'].iloc[0], group['min_val_loss'].iloc[1]
            param_val1, param_val2 = group[param].iloc[0], group[param].iloc[1]
            
            arrow_color = 'black'
            ax.arrow(x1, y_pos, x2-x1, 0, head_width=0.1, head_length=0.005, 
                    length_includes_head=True, color=arrow_color)
            
            ax.plot(x1, y_pos, 'o', color='blue', 
                   label=f'{param}={param_val1}' if y_pos==0 else "")
            ax.plot(x2, y_pos, 'o', color='red', 
                   label=f'{param}={param_val2}' if y_pos==0 else "")
            
            # Store y-axis info
            y_ticks.append(y_pos)
            if isinstance(combo, tuple):
                label = ', \n'.join(f'{p}={v}' for p,v in zip(other_params, combo))
            else:
                label = f'{other_params[0]}={combo}'
            y_labels.append(label)
            y_pos += 1
            
        # Add y-axis labels back
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Validation Loss')
        ax.set_title(f'Loss Change by {param}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

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
def check_2d_reconstructions(x, x_reconstructed):
    # Check that reconstructions are approximately the same as the inputs
    eps = 1e-4
    are_close = torch.allclose(x, x_reconstructed, atol = eps)
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
    

    plt.scatter(z[:, 0], z[:, 1], 
            c='blue', marker='o')
    
    plt.title(f'Code distribution: MMD^2 = {mmd2}, \nvar = {torch.var(z)}, mean = {z.mean()}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def check_generation_quality(x_synth, x_test, quality, bandwidth_start = 0.5, num_bandwidths = 5):
    mmd2 = MMD2(x_synth, x_test, bandwidth_start, num_bandwidths)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test[:, 0], x_test[:, 1], 
            c='blue', marker='o', label='Test Data')
    plt.scatter(x_synth[:, 0], x_synth[:, 1], 
            c='red', marker='x', label='Synthetic Data')

    plt.title(f'Test vs Synthetic Points ({quality} quality: MMD^2 = {mmd2})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_analysis(best_path, bad_path, best_params, worst_params, dataset, noise = 0.1, seed = 11121, n_test = 100, test_batch_size = 32):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Get models
    input_size = 2
    good_inn = RealNVP(input_size, int(best_params['hidden_size']), int(best_params['blocks']), 'cpu')
    checkpoint = torch.load(best_path, weights_only=False)
    good_inn.load_state_dict(checkpoint['model_state_dict'])

    bad_inn = RealNVP(input_size, int(worst_params['hidden_size']), int(worst_params['blocks']), 'cpu')
    checkpoint = torch.load(bad_path, weights_only=False)
    bad_inn.load_state_dict(checkpoint['model_state_dict'])

    # Get various datasets
    if dataset == 'moons':
        x_test = get_standardised_moons(int(best_params['n_train']), noise)
    elif dataset == 'gmm':
        x_test = get_standardised_gmm(int(best_params['n_train']), 1/10)
    z_test = good_inn.get_codes(x_test, test_batch_size) # Codes corresponding to test data
    x_reconstructed_test = good_inn.get_reconstructions(z_test, test_batch_size) # Reconstructions of test data
    x_synth_good = good_inn.sample(n_test, seed) # good synthetic data
    x_synth_bad = bad_inn.sample(n_test, seed) # bad synthetic data

    # Run checks
    check_2d_reconstructions(x_test, x_reconstructed_test)
    check_normality(z_test)

    check_generation_quality(x_synth_good, x_test, quality = 'good')
    check_generation_quality(x_synth_bad, x_test, quality = 'bad')   

if __name__ == '__main__':
    # Hyperparam analysis
    min_losses_df = pd.read_csv('ex3/min_losses_moons.csv')

    # Tabular analysis: Which is best model
    best_path, worst_path, best_params, worst_params = find_best_model_from_tabular(min_losses_df)

    # How do hparams change performance
    plot_training_histories(best_path, worst_path, model_names=['Best Model', 'Worst Model'])
    plot_all_parameter_changes(min_losses_df)

    run_analysis(best_path, worst_path, best_params, worst_params, 'moons')

    # Rerun analysis for gmm

    # Tabular analysis: Which is best model
    best_path, worst_path, best_params, worst_params = find_best_model_from_tabular(min_losses_df)

    run_analysis(best_path, worst_path, best_params, worst_params, 'gmm')
    