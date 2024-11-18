import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from functools import partial
from AE import AutoEncoder, train_model

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
            result[i, j] = torch.pow((torch.sum(torch.power(a[i]-b[j], 2))/h+1), -1)  
    return result

def multi_bandwidth_kernel(a, b, kernel, bandwidths):
    """
    Given a kernel with a bandwidth parameter, return the sum of the kernel values for each bandwidth
    """
    bandwidthed_kernel_values = torch.zeros(len(bandwidths))
    for i, h in enumerate(bandwidths):
        bandwidthed_kernel_values[i] = kernel(a, b, h)

    return bandwidthed_kernel_values.sum()

def MMD2(prediction_set, true_set, bandwidth_start, num_bandwidths):
    bandwidths = [bandwidth_start*2**i for i in range(num_bandwidths)]
    # Define kernel functions to use: Sum of squared exponential kernels or 
    # sum of inverse multiquadratic kernels.
    kernels = [
        partial(multi_bandwidth_kernel, 
                kernel = squared_exponential_kernel, 
                bandwidths = bandwidths),
        partial(multi_bandwidth_kernel, 
                kernel = inverse_multiquad_kernel, 
                bandwidths = bandwidths)
    ]


    # Convert to np array if needed
    if isinstance(prediction_set, list):
        prediction_set = np.array(prediction_set)
    if isinstance(true_set, list):
        true_set = np.array(true_set)

    # Find number of predictions and true samples
    N = prediction_set.shape[0]
    M = true_set.shape[0]

    # Calculate squared mean discrepancy per kernel and return maximum
    squared_mean_discrepancies = []
    for kernel in kernels:
        
        kernel_squared_mean_discrepancy = \
            (np.sum(kernel(true_set, true_set)) - np.sum(np.diag(kernel(true_set, true_set))))/(N*(N-1)) + \
            (np.sum(kernel(prediction_set, prediction_set))-np.sum(np.diag(kernel(prediction_set, prediction_set))))/(M*(M-1)) - \
            2*(np.sum(kernel(prediction_set, true_set)))/(N*M)
        
        squared_mean_discrepancies.append(kernel_squared_mean_discrepancy)
    
    mmd2 = squared_mean_discrepancies.max()

    return mmd2

def mmd2_mse_loss(x_hat, code, x_batch, bandwidth_start, num_bandwidths, mmd_weight):
    # Get MSE loss
    mse = nn.functional.mse_loss(x_hat, x_batch)

    # Get MMD2 loss between some random normal samples and the codes.
    true_set = torch.randn(x_hat.shape)
    mmd2 = MMD2(code, true_set, bandwidth_start, num_bandwidths)
    
    return mse + mmd_weight*mmd2

def train_mmd_autoencoder(autoencoder, n_epoch, loss_fn, x_standardised, lr):
    return train_model(autoencoder, n_epoch, loss_fn, x_standardised, lr, mmd_loss = True)

if __name__ == '__main__':
    # Hparams
    ## Architecture hparams
    input_size = 2 # Data dimensionaliy
    bottleneck_size = 2
    hidden_size = 2 # Maximum dimension of hidden layers
    layers = 3 # number of layers in the encoder

    ## Training hparams
    n_train = 1000
    lr = 0.001
    n_epoch = 1000
    seed = 11121
    mmd_weight = 1
    bandwidth_start = 0.5
    num_bandwidths = 5

    # Get data
    x_train = datasets.make_moons(n_samples = n_train, noise = 0.1)[0]

    ## convert data to tensor and standardise
    x_train = torch.from_numpy(x_train).float()
    mean, sd = x_train.mean(dim=0), x_train.std(dim=0)
    x_standardised = (x_train-mean)/sd

    # Get model
    if seed is not None:
        torch.manual_seed(seed)

    autoencoder = AutoEncoder(input_size, bottleneck_size, hidden_size, layers)

    # Train model
    loss_fn = partial(mmd2_mse_loss, bandwidth_start = bandwidth_start, num_bandwidths = num_bandwidths, mmd_weight = mmd_weight)
    train_mmd_autoencoder(autoencoder, n_epoch, loss_fn, x_standardised, lr)