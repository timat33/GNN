from itertools import product
import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from argparse import Namespace
import pandas as pd
import os

# Data functions
def get_standardised_moons(n, noise = 0.1, device = 'cpu'):
    x = datasets.make_moons(n_samples = n, noise = noise)[0]

    ## convert data to tensor and standardise
    x = torch.from_numpy(x).float().to(device)
    mean, sd = x.mean(dim=0), x.std(dim=0)
    x_standardised = (x-mean)/sd

    return x_standardised

def get_standardised_gmm(n_samples, radius, device = 'cpu'):
    # Get vertices of regular hexagon centered at origin with radius radius
    thetas = 2*np.pi/6 * np.arange(6)
    vertices = np.stack([
        radius * np.cos(thetas),
        radius * np.sin(thetas)
    ]).T  # Better stacking for 2D points
    
    # Smaller covariance for tighter clusters
    covariance_matrix = np.eye(2) * (radius/10)**2  # Square for proper variance scale
    covs = np.array([covariance_matrix for _ in range(6)])
    precisions = np.linalg.inv(covs)
    
    # Setup GMM
    gmm = GaussianMixture(n_components=6, covariance_type='full')
    gmm.means_ = vertices
    gmm.covariances_ = covs
    gmm.weights_ = np.ones(6)/6
    gmm.precisions_cholesky_ = np.linalg.cholesky(precisions).transpose(0, 2, 1)
    gmm.converged_ = True
    gmm.n_iter_ = 0
    
    # Generate samples
    x, _ = gmm.sample(n_samples)

    ## convert data to tensor and standardise
    x = torch.from_numpy(x).float().to(device)
    mean, sd = x.mean(dim=0), x.std(dim=0)
    x_standardised = (x-mean)/sd
    
    return x_standardised

# Normalising flow (INN) architecture and helper classes
def get_rand_rotation_mat(n):
    '''
    Obtain random rotation matrix from qr decomposition of a standard normal array
    '''
    a = np.random.randn(n, n)
    q, _ = np.linalg.qr(a)
    return q

class translation_net(nn.Module):
    def __init__(self, in_features, out_features, width):
        '''
        in_features: int, number of input features
        width: int, width of the intermediate layers
        Take input of size in_features and output a translation value as follows:
        take input, pass through two fully connected layers with ReLU activation, 
        then pass through another fully connected layer and output a single value
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width

        self.fc1 = nn.Linear(in_features, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, self.out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        intermediate1 = self.relu(self.fc1(x))
        intermediate2 = self.relu(self.fc2(intermediate1))
        output = self.fc3(intermediate2)
        return output
    
class scaling_net(nn.Module):
    def __init__(self, in_features, out_features, width):
        '''
        in_features: int, number of input features
        width: int, width of the intermediate layers
        Same as translation block, but pass output through exp(tanh)
        '''
        super().__init__()
        # Store parameters
        self.in_features = in_features
        self.out_features = out_features
        self.width = width

        # Layers
        self.fc1 = nn.Linear(in_features, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, self.out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        intermediate1 = self.relu(self.fc1(x))
        intermediate2 = self.relu(self.fc2(intermediate1))
        output_unfixed = self.fc3(intermediate2)
        output = torch.exp(torch.tanh(output_unfixed))
        return output
    
class coupling_layer(nn.Module):
    def __init__(self, data_dim, width):
        '''
        Coupling block; as described in lecture
        '''
        super().__init__()
        # Store parameters
        self.data_dim = data_dim
        self.width = width
        self.D_tilde = data_dim // 2 # Number of features to skip
        self.coupling_output_dim = data_dim - self.D_tilde # same as self.D_tilde if data_dim is even

        # Subnetworks
        self.translation = translation_net(self.D_tilde, self.coupling_output_dim, width)
        self.scaling = scaling_net(self.D_tilde, self.coupling_output_dim, width)

    def forward(self, x):
        # Obtain scaling and translation coeffs
        trans_coeff = self.translation(x[:, :self.D_tilde]) # all batches; first half
        scale_coeff = self.scaling(x[:, :self.D_tilde]) 

        # Skip features up to index D_tilde, apply scaling and translation to the rest element-wise
        output = torch.cat([x[:, :self.D_tilde], # Skip connection
                            x[:, self.D_tilde:] * scale_coeff + trans_coeff], # Transformed
                            dim = 1) # concatenate for each batch element
        return output
    
    def reverse(self, x):
        # Obtain scaling and translation coeffs
        trans_coeff = self.translation(x[:, :self.D_tilde]) # all batches; first half
        scale_coeff = self.scaling(x[:, :self.D_tilde]) 
        

        # Skip features up to index D_tilde, apply scaling and translation to the rest element-wise
        output = torch.cat([x[:, :self.D_tilde], # Skip connection
                            (x[:, self.D_tilde:] - trans_coeff)/scale_coeff], # Transformed
                            dim = 1) # concatenate for each batch element

        return output

class RealNVP(nn.Module):
    def __init__(self, input_size, hidden_size, blocks, device = 'cpu'):
        '''
        input_size: dimension of input data
        hidden_size: width of subnetworks that determine scaling and translation
        blocks: number of coupling layers in the model
        
        Proceed through `blocks` number of coupling layers with subnetworks of width `hidden_size`
        '''
        super().__init__()
        # Store parameters
        self.data_dim = input_size
        self.hidden_size = hidden_size
        self.blocks = blocks

        self.device = device

        # Create coupling layers
        self.coupling_layers = nn.ModuleList(
            [coupling_layer(self.data_dim, self.hidden_size) for _ in range(self.blocks)]
        )

        # Get rotation matrices
        self.rotation_matrices = nn.ParameterList([
            nn.Parameter(
                torch.tensor(
                    get_rand_rotation_mat(self.data_dim),
                    dtype=torch.float32,
                    device=self.device
                ),
                requires_grad=False
            ) for _ in range(self.blocks-1)
        ])


    def forward(self, x):
        # Apply coupling layers, interspersing with rotation matrices. Store intermediate output halves
        z = x
        intermediates = []
        for i, coupling_layer in enumerate(self.coupling_layers):
            # Store intermediate half for loss calculation
            intermediates.append(z)

            # pass through coupling layer
            z = coupling_layer(z) 

            if i!= self.blocks-1:
                # apply rotation matrix (except for last layer)
                z = torch.einsum('ij,bj->bi', self.rotation_matrices[i], z)
                
        return intermediates, z

    def reverse(self, z):
        # Apply coupling layers in reverse order, interspersing with inverse rotation matrices
        x_hat = z
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            if i!= self.blocks-1:
                # apply inverse rotation matrix (except for first layer)
                x_hat = torch.einsum('ij,bj->bi', 
                                      torch.inverse(self.rotation_matrices[i]), 
                                      x_hat)

            # pass through coupling layer
            x_hat = coupling_layer.reverse(x_hat)

        return x_hat

    # Inference functions
    def get_codes(self, x, batch_size):
        '''
        Pass a test tensor of data points through the mode
        '''
        test_loader = DataLoader(x, batch_size = batch_size)

        outputs = []
        with torch.no_grad():
            for x_batch in test_loader:
                x_batch = x_batch.to(self.device)
                _, output = self.forward(x_batch) # Discard intermediates
                outputs.append(output)

        z = torch.cat(outputs, dim = 0) 

        return z
    
    def get_reconstructions(self, z, batch_size):
        reverse_loader = DataLoader(z, batch_size = batch_size)

        outputs = []
        with torch.no_grad():
            for x_batch in reverse_loader:
                x_batch = x_batch.to(self.device)
                output = self.reverse(x_batch)
                outputs.append(output)

        x_reconstructed = torch.cat(outputs, dim = 0)

        return x_reconstructed
    
    def sample(self, n, batch_size = 32, seed = 11121):
        """
        Sample from data distribution by generating normal samples and passing through
        the model in reverse.
        """
        if seed:
            torch.manual_seed(seed)

        codes = torch.randn(n,self.data_dim)
        reconstructions = self.get_reconstructions(codes, batch_size)

        return reconstructions

# Functions for training
class NLLLoss(nn.Module):
    def __init__(self, coupling_layers, data_dim):
        super().__init__()
        self.coupling_layers = coupling_layers
        self.data_dim = data_dim
    
    def forward(self, intermediates, z):
        # Component corresponding to transforming the data
        transformed_component = (z**2).sum(dim=1)/2
        
        # Get log_det_loss component
        log_det_component = 0
        for i in range(len(intermediates)):
            # Get z_{\leq D_tilde}
            lower_half = intermediates[i][:, :self.data_dim//2]
            
            # Get sum of scaling coefficients
            scaling_sum = torch.log(self.coupling_layers[i].scaling(lower_half)).sum(dim=1)
            log_det_component += scaling_sum
        
        return (transformed_component - log_det_component).mean()
    
def train_epoch(model: RealNVP, train_loader, optimiser, loss_fn):
    model.train()
    train_loss = 0.

    for i, x_batch in enumerate(train_loader):
        # Move to gpu if possible
        x_batch = x_batch.to(model.device)

        # Zero out gradients
        optimiser.zero_grad()

        # Make predictions
        intermediates, x_hat = model(x_batch)

        # Get loss, metrics, and gradients
        loss = loss_fn(intermediates, x_hat)
        loss.backward()

        # Update
        optimiser.step()

        # Track loss
        train_loss += loss.item()

    # Get loss averaged over batches
    train_loss = train_loss/len(train_loader)


    return train_loss

def get_train_val_split(x_train, batch_size, seed = 11121):
    # Make a training/validation split: shuffle and then split 80/20
    if seed:
        torch.manual_seed(seed)

    shuffled_inds = torch.randperm(x_train.shape[0])
    x_train = x_train[shuffled_inds]
    cut_index = np.floor(0.8*x_train.shape[0]).astype(int)
    x_train, x_val = x_train[:cut_index], x_train[cut_index:]

    train_loader = DataLoader(x_train, batch_size = batch_size)
    val_loader = DataLoader(x_val, batch_size=batch_size)

    return train_loader, val_loader

@torch.no_grad()
def get_val_loss(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    for x_batch in val_loader:
        # Move to gpu if possible
        x_batch = x_batch.to(model.device)

        # Make predictions
        intermediates, x_hat = model(x_batch)

        # Get loss
        loss = loss_fn(intermediates, x_hat)

        val_loss+=loss.item()

    # Divide to get loss averaged over batches
    val_loss = val_loss/len(val_loader)

    return val_loss

def train_model(model, n_epoch, loss_fn, x_train, lr, model_path, batch_size = 4, seed = 11121, device = 'cpu'):
    # Get extra necessary objects
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Move to gpu if possible
    x_train = x_train.to(device)

    # Define counter for early stopping to avoid overfitting/computation inefficiency
    early_stop_counter = 0
    early_stop_counter_max = 15 # Stop if no improvement in val loss after this many epochs

    # Make a training/validation split: shuffle and then split
    train_loader, val_loader = get_train_val_split(x_train, batch_size, seed)

    # Train. Terminate early based on validation loss
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch_index in range(n_epoch):

        model.train()
        # Train and get validation loss
        train_loss = train_epoch(model, train_loader, optimiser, loss_fn)

        val_loss = get_val_loss(model, val_loader, loss_fn)

        # print outputs
        if epoch_index % 10 == 0:
            print('EPOCH {}:'.format(epoch_index + 1))
            print(f'  training batch loss: {train_loss}')
            print(f'  validation batch loss: {val_loss}')

        # Store losses
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # If best loss is beat, then keep going. Else increment counter. Stop if counter gets too high
        if val_loss < best_val_loss:
            early_stop_counter = 0 
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': best_val_loss,
                'history': history
            }, model_path)

        else:
            early_stop_counter +=1
            if early_stop_counter == early_stop_counter_max:
                print(f'Early stopping after {epoch_index + 1} epochs')
                break

    return history

def init_and_train(hparams, fixed_params, model_path, dataset):
    # Get data
    if dataset == 'moons':
        x_standardised = get_standardised_moons(hparams.n_train, fixed_params.noise, fixed_params.device)
    elif dataset == 'gmm':
        x_standardised = get_standardised_gmm(hparams.n_train, fixed_params.noise, fixed_params.device)

    # Get model
    if fixed_params.seed is not None:
        torch.manual_seed(fixed_params.seed)

    inn = RealNVP(fixed_params.input_size, 
                  hparams.hidden_size, 
                  hparams.blocks, 
                  fixed_params.device).to(fixed_params.device)

    # Train model
    loss_fn = NLLLoss(inn.coupling_layers, fixed_params.input_size)
    history = train_model(inn, hparams.n_epoch, loss_fn, x_standardised, hparams.lr, model_path=model_path, device = fixed_params.device, batch_size=fixed_params.batch_size)

    return history

def init_and_train_from_grid(hparams_grid, fixed_params, model_path_template, dataset):
    '''
    Take grid of hyperparams and train models for all combinations
    '''
    # Copy hyperparams
    hparams = Namespace(**vars(hparams_grid))
    results = pd.DataFrame(columns=['hidden_size', 'blocks', 'n_train', 'lr', 'min_val_loss'])
    
    for hidden_size, blocks, n_train, lr in product(hparams_grid.hidden_size, hparams_grid.blocks, hparams_grid.n_train, hparams_grid.lr):
    # Overwrite list with fixed value
        hparams.n_train = n_train
        hparams.hidden_size = hidden_size
        hparams.blocks = blocks
        hparams.lr = lr

        model_path = model_path_template.replace('.pt', 
                                       f'_ntrain{n_train}_hiddensize{hidden_size}_blocks{blocks}_lr{str(lr).replace('.',',')}.pt')
        print(f'Training model {model_path}')

        history = init_and_train(hparams, fixed_params, model_path, dataset)

        # Get minimum validation loss
        min_val_loss = min(history['val_loss'])
        
        # Add row to results DataFrame
        results.loc[len(results)] = [int(hidden_size), int(blocks), int(n_train), lr, min_val_loss]
    
    return results

if __name__ == '__main__':
    # Hparams
    hparams_grid = Namespace()
    fixed_params = Namespace()

    ## Architecture hparams
    hparams_grid.hidden_size = [16,24] 
    hparams_grid.blocks = [12,18]

    ## Training hparams
    hparams_grid.n_train = [1000,2000]
    hparams_grid.lr = [0.01,0.02]
    hparams_grid.n_epoch = 200

    # Fixed params
    fixed_params.input_size = 2 
    fixed_params.batch_size = 16
    fixed_params.noise = 0.1
    fixed_params.seed = 11121
    fixed_params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Apply to moons dataset
    # os.makedirs('ex3/models/moons', exist_ok=True)
    # best_model_path=None #'models/moons/moons_INN.pt' # For safety
    # min_losses = init_and_train_from_grid(hparams_grid, fixed_params, best_model_path, 'moons')

    # min_losses.to_csv('min_losses_moons.csv', index=False)

    # Apply to gmm dataset
    os.makedirs('ex3/models/gmms', exist_ok=True)
    best_model_path='ex3/models/gmms/gmms_INN.pt'
    min_losses = init_and_train_from_grid(hparams_grid, fixed_params, best_model_path, 'gmm')

    # Save results
    min_losses.to_csv('ex3/min_losses_gmm.csv', index=False)

