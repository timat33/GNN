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

def mmd2_mse_loss(x_hat, code, x_batch, bandwidth_start, num_bandwidths, mmd_weight):
    # Get MSE loss
    mse = nn.functional.mse_loss(x_hat, x_batch)

    # Get MMD2 loss between some random normal samples and the codes.
    true_set = torch.randn(x_hat.shape)
    mmd2 = MMD2(code, true_set, bandwidth_start, num_bandwidths)
    
    return mse + mmd_weight*mmd2

def train_mmd_autoencoder(autoencoder, n_epoch, loss_fn, x_standardised, lr):
    return train_model(autoencoder, n_epoch, loss_fn, x_standardised, lr, mmd_loss = True)

def gen_synth_data(autoencoder, n_synth, mean, sd):
    autoencoder.eval()
    codes = torch.randn(n_synth, autoencoder.bottleneck_size)
    synth_data = autoencoder.decoder(codes)

    # Undo normalisation
    synth_data = synth_data*sd + mean
    return synth_data

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
    n_epoch = 50
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
    autoencoder.eval()
    
    # Generate synthetic data
    n_synth = 10
    synth_data = gen_synth_data(autoencoder, n_synth, mean, sd)
    
