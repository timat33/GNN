from itertools import product
from typing import List
import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from argparse import Namespace
import pandas as pd
import os

# Data functions
def get_standardised_moons(n, conditional, noise = 0.1, device = 'cpu'):
    x, labels = datasets.make_moons(n_samples = n, noise = noise)

    ## convert data to tensor and standardise
    x = torch.from_numpy(x).float().to(device)
    mean, sd = x.mean(dim=0), x.std(dim=0)
    x_standardised = (x-mean)/sd

    # If conditional, keep labels. Else discard
    if conditional: 
        labels = torch.from_numpy(labels).to(device)
    else:
        labels = None

    return x_standardised, labels

def relabel_to_binary(labels: torch.Tensor, positive_labels: List = [0,2]) -> torch.Tensor:
    """
    Relabel label tensor so 0,2 -> 0 and all else -> 1
    """
    new_labels = torch.zeros_like(labels)

    # Find indices where labels are in positive_labels
    mask = torch.isin(labels, torch.tensor(positive_labels))
    new_labels[mask] = 1
    return new_labels

def get_standardised_gmm(n_samples, radius, conditional, device = 'cpu'):
    # Get vertices of regular hexagon centered at origin with radius radius
    thetas = 2*np.pi/6 * np.arange(6)
    vertices = np.array([np.cos(thetas), np.sin(thetas)]).reshape(6,2)

    covariance_matrix = np.eye(2)*radius/10
    covs = np.array([covariance_matrix for _ in range(6)])  # Shape: (6,2,2)
    
    
    # Create GMM
    gmm = GaussianMixture(
        n_components=6,
        covariance_type='full',
        weights_init=np.ones(6)/6  # equal weights
    )
    
    # Set parameters manually
    gmm.means_ = vertices
    gmm.covariances_ = covs
    gmm.weights_ = np.ones(6)/6
    gmm.precisions_cholesky_ = np.linalg.cholesky(
        np.linalg.inv(covs)
    ).transpose(0, 2, 1)
    
    # Generate samples
    x, labels = gmm.sample(n_samples)

    ## convert data to tensor and standardise
    x = torch.from_numpy(x).float().to(device)
    mean, sd = x.mean(dim=0), x.std(dim=0)
    x_standardised = (x-mean)/sd

    # If conditional, keep labels. Else discard
    if conditional: 
        labels = torch.from_numpy(labels).to(device)
    else:
        labels = None
    
    return x_standardised, labels

# Normalising flow (INN) architecture and helper classes
def get_rand_rotation_mat(n):
    '''
    Obtain random rotation matrix from qr decomposition of a standard normal array
    '''
    a = np.random.randn(n, n)
    q, _ = np.linalg.qr(a)
    return q

class translation_net(nn.Module):
    def __init__(self, in_features, out_features, width, condition_size = 0):
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
        self.condition_size = condition_size

        self.fc1 = nn.Linear(in_features + condition_size, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, self.out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        intermediate1 = self.relu(self.fc1(x))
        intermediate2 = self.relu(self.fc2(intermediate1))
        output = self.fc3(intermediate2)
        return output
    
class scaling_net(nn.Module):
    def __init__(self, in_features, out_features, width, condition_size):
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
        self.conditional_size = condition_size

        # Layers
        self.fc1 = nn.Linear(in_features + condition_size, width)
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
    def __init__(self, data_dim, width, condition_size):
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
        self.translation = translation_net(self.D_tilde, self.coupling_output_dim, width, condition_size)
        self.scaling = scaling_net(self.D_tilde, self.coupling_output_dim, width, condition_size)

    def forward(self, x, condition = None):
        lower_half = x[:, :self.D_tilde]
        upper_half = x[:, self.D_tilde:]
        
        # Obtain scaling and translation coeffs
        if condition is None:
            # Use lower_half directly without conditions
            trans_coeff = self.translation(lower_half)
            scale_coeff = self.scaling(lower_half)
        else:
            # Add conditions
            lower_half_with_conditions = torch.cat([
                lower_half,
                condition.unsqueeze(1)
            ], dim=1)
            trans_coeff = self.translation(lower_half_with_conditions)
            scale_coeff = self.scaling(lower_half_with_conditions)

        # Skip features up to index D_tilde, apply scaling and translation to the rest element-wise
        output = torch.cat([lower_half, # Skip connection
                            upper_half * scale_coeff + trans_coeff], # Transformed
                            dim = 1) # concatenate for each batch element
        return output
    
    def reverse(self, x, condition):
        lower_half = x[:, :self.D_tilde]
        upper_half = x[:, self.D_tilde:]
        
        # Obtain scaling and translation coeffs
        if condition is None:
            # Use lower_half directly without conditions
            trans_coeff = self.translation(lower_half)
            scale_coeff = self.scaling(lower_half)
        else:
            # Add conditions
            lower_half_with_conditions = torch.cat([
                lower_half,
                condition.unsqueeze(1)
            ], dim=1)
            trans_coeff = self.translation(lower_half_with_conditions)
            scale_coeff = self.scaling(lower_half_with_conditions)

        # Skip features up to index D_tilde, apply scaling and translation to the rest element-wise
        output = torch.cat([lower_half, # Skip connection
                            (upper_half - trans_coeff)/scale_coeff], # Transformed
                            dim = 1) # concatenate for each batch element

        return output

class RealNVP(nn.Module):
    def __init__(self, input_size, hidden_size, blocks, condition_size, device = 'cpu'):
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
        self.condition_size = condition_size

        # Create coupling layers
        self.coupling_layers = nn.ModuleList(
            [coupling_layer(self.data_dim, self.hidden_size, condition_size) for _ in range(self.blocks)]
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


    def forward(self, x, condition = None):
        # Apply coupling layers, interspersing with rotation matrices. Store intermediate output halves
        z = x
        intermediates = []
        for i, coupling_layer in enumerate(self.coupling_layers):
            # Store intermediate half for loss calculation
            intermediates.append(z)

            # pass through coupling layer
            z = coupling_layer(z, condition) 

            if i!= self.blocks-1:
                # apply rotation matrix (except for last layer)
                z = torch.einsum('ij,bj->bi', self.rotation_matrices[i], z)
                
        return intermediates, z

    def reverse(self, z, condition):
        # Apply coupling layers in reverse order, interspersing with inverse rotation matrices
        x_hat = z
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            if i!= self.blocks-1:
                # apply inverse rotation matrix (except for first layer)
                x_hat = torch.einsum('ij,bj->bi', 
                                      torch.inverse(self.rotation_matrices[i]), 
                                      x_hat)

            # pass through coupling layer
            x_hat = coupling_layer.reverse(x_hat, condition)

        return x_hat

    # Inference functions
    @torch.no_grad()
    def get_codes(self, x, conditions=None, batch_size=32):
        '''
        Pass a test tensor of data points through the mode
        '''
        outputs = []
        if conditions is None:
            test_loader = DataLoader(x, batch_size = batch_size)


            for x_batch, in test_loader:
                x_batch = x_batch.to(self.device)
                _, output = self.forward(x_batch) # Discard intermediates
                outputs.append(output)
        else:
            test_dataset = TensorDataset(x, conditions)
            test_loader = DataLoader(test_dataset, batch_size = batch_size)


            for x_batch, labels_batch in test_loader:
                x_batch = x_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                _, output = self.forward(x_batch, labels_batch) # Discard intermediates
                outputs.append(output)

        z = torch.cat(outputs, dim = 0) 

        return z
    
    @torch.no_grad()
    def get_reconstructions(self, z, condition = None, batch_size = 32,):
        
        if condition is None:
            reverse_loader = DataLoader(z, batch_size = batch_size)

            outputs = []
            for x_batch in reverse_loader:
                x_batch = x_batch.to(self.device)
                output = self.reverse(x_batch, condition)
                outputs.append(output)

        else:
            reverse_dataset = TensorDataset(z, condition)
            reverse_loader = DataLoader(reverse_dataset, batch_size = batch_size)

            outputs = []
            for x_batch, condition_batch in reverse_loader:
                x_batch = x_batch.to(self.device)
                condition_batch = condition_batch.to(self.device)

                output = self.reverse(x_batch, condition_batch)
                outputs.append(output)

        x_reconstructed = torch.cat(outputs, dim = 0)

        return x_reconstructed
    
    def sample(self, n, batch_size = 32, conditions = None, seed = 11121):
        """
        Sample from data distribution by generating normal samples and passing through
        the model in reverse.
        """
        if seed:
            torch.manual_seed(seed)

        # If no conditions, just generate sample. Else, generate sample conditioned on conditions
        if conditions is None:
            codes = torch.randn(n,self.data_dim)
            reconstructions = self.get_reconstructions(codes, batch_size = batch_size)
            labels = None
        else: 
            if not isinstance(conditions, torch.Tensor):
                conditions = torch.tensor(conditions).to(self.device)
            
            # Create n samples for each condition then concatenate
            reconstructions = []
            labels = []
            
            for condition in conditions:
                codes = torch.randn(n,self.data_dim)
                contitional_labels = torch.ones(n)*condition # Each sample has the same condition
                conditional_reconstructions = self.get_reconstructions(codes, contitional_labels, batch_size)
                
                
                reconstructions.append(conditional_reconstructions)
                labels.append(contitional_labels)

            reconstructions = torch.cat(reconstructions, dim = 0)
            labels = torch.cat(labels, dim = 0)


        return reconstructions, labels

# Functions for training
class NLLLoss(nn.Module):
    def __init__(self, coupling_layers, data_dim):
        super().__init__()
        self.coupling_layers = coupling_layers
        self.data_dim = data_dim
    
    def forward(self, intermediates, z, condition):
        # Component corresponding to transforming the data
        transformed_component = (z**2).sum(dim=1)/2
        
        # Get log_det_loss component
        log_det_component = 0
        for i in range(len(intermediates)):
            # Get z_{\leq D_tilde}
            lower_half = intermediates[i][:, :self.data_dim//2]
            
            # Get sum of scaling coefficients
            if condition is None:
                # Use lower_half directly without conditions
                scale_coeff = self.coupling_layers[i].scaling(lower_half)
            else:
                # Add conditions
                lower_half_with_conditions = torch.cat([
                    lower_half,
                    condition.unsqueeze(1)
                ], dim=1)
                scale_coeff = self.coupling_layers[i].scaling(lower_half_with_conditions)
            scaling_sum = torch.log(scale_coeff).sum(dim=1)
            log_det_component += scaling_sum
        
        return (transformed_component - log_det_component).mean()
    
def train_epoch(model: RealNVP, train_loader, optimiser, loss_fn: NLLLoss, conditional: bool):
    model.train()
    train_loss = 0.

    for x_batch, label_batch in train_loader:
        # Move to gpu if possible
        x_batch, label_batch = x_batch.to(model.device), label_batch.to(model.device)
        
        if not conditional:
            label_batch = None

        # Zero out gradients
        optimiser.zero_grad()

        # Make predictions
        intermediates, x_hat = model(x_batch, label_batch)

        # Get loss, metrics, and gradients
        loss = loss_fn(intermediates, x_hat, label_batch)
        loss.backward()

        # Update
        optimiser.step()

        # Track loss
        train_loss += loss.item()

    # Get loss averaged over batches
    train_loss = train_loss/len(train_loader)


    return train_loss


def get_train_val_split(x_train, labels, batch_size, seed=11121):
    """Split data and corresponding labels into train/val sets"""
    if seed:
        torch.manual_seed(seed)

    if labels is None:
        labels = torch.zeros(x_train.shape[0]) # These will not be used anywhere.

    # Get shuffled indices and apply to both tensors
    shuffled_inds = torch.randperm(x_train.shape[0])
    x_train = x_train[shuffled_inds]
    labels = labels[shuffled_inds]

    # Split 80/20
    cut_index = int(np.floor(0.8 * x_train.shape[0]))
    x_train, x_val = x_train[:cut_index], x_train[cut_index:]
    labels_train, labels_val = labels[:cut_index], labels[cut_index:]

    # Create datasets of pairs
    train_dataset = TensorDataset(x_train, labels_train)
    val_dataset = TensorDataset(x_val, labels_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

@torch.no_grad()
def get_val_loss(model, val_loader, loss_fn, conditional):
    model.eval()
    val_loss = 0
    for x_batch, label_batch in val_loader:
        # Move to gpu if possible
        x_batch, label_batch = x_batch.to(model.device), label_batch.to(model.device)

        if not conditional:
            label_batch = None

        # Make predictions
        intermediates, x_hat = model(x_batch, label_batch)

        # Get loss
        loss = loss_fn(intermediates, x_hat, label_batch)

        val_loss+=loss.item()

    # Divide to get loss averaged over batches
    val_loss = val_loss/len(val_loader)

    return val_loss

def train_model(model, n_epoch, loss_fn, x_train, labels, conditional, lr, model_path, batch_size = 4, seed = 11121, device = 'cpu'):
    # Get extra necessary objects
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Move to gpu if possible
    x_train = x_train.to(device)

    # Define counter for early stopping to avoid overfitting/computation inefficiency
    early_stop_counter = 0
    early_stop_counter_max = 15 # Stop if no improvement in val loss after this many epochs

    # Make a training/validation split: shuffle and then split
    train_loader, val_loader = get_train_val_split(x_train, labels, batch_size, seed)

    # Train. Terminate early based on validation loss
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch_index in range(n_epoch):

        model.train()
        # Train and get validation loss
        train_loss = train_epoch(model, train_loader, optimiser, loss_fn, conditional)

        val_loss = get_val_loss(model, val_loader, loss_fn, conditional)

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

def init_and_train(hparams, fixed_params, model_path, dataset, conditional: bool):
    # Get data
    if dataset == 'moons':
        x_standardised, labels = get_standardised_moons(hparams.n_train, fixed_params.conditional, fixed_params.noise, fixed_params.device)
    elif dataset == 'gmm':
        x_standardised, labels = get_standardised_gmm(hparams.n_train, fixed_params.conditional, fixed_params.noise, fixed_params.device)
        labels = relabel_to_binary(labels, positive_labels=fixed_params.positive_labels)

    if not conditional:
        labels = None

    # Get model
    if fixed_params.seed is not None:
        torch.manual_seed(fixed_params.seed)

    inn = RealNVP(fixed_params.input_size, 
                  hparams.hidden_size, 
                  hparams.blocks, 
                  fixed_params.condition_size,
                  fixed_params.device).to(fixed_params.device)

    # Train model
    loss_fn = NLLLoss(inn.coupling_layers, fixed_params.input_size)
    history = train_model(inn, hparams.n_epoch, loss_fn, x_standardised, labels, fixed_params.conditional, hparams.lr, model_path=model_path, device = fixed_params.device, batch_size=fixed_params.batch_size)

    return history

def init_and_train_from_grid(hparams_grid, fixed_params, model_path_template, dataset, conditional):
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

        history = init_and_train(hparams, fixed_params, model_path, dataset, conditional)

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
    fixed_params.condition_size = 1
    fixed_params.conditional = True
    fixed_params.batch_size = 16
    fixed_params.noise = 0.1
    fixed_params.seed = 11121
    fixed_params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_params.positive_labels = [0,2] # For gmm dataset

    # Apply to moons dataset
    os.makedirs('models/moons_conditional', exist_ok=True)
    best_model_path='ex3/models/moons_conditional/moons_INN.pt'
    min_losses = init_and_train_from_grid(hparams_grid, fixed_params, best_model_path, 'moons', conditional = True)

    min_losses.to_csv('min_losses_conditional_moons.csv', index=False)

    # Apply to gmm dataset
    os.makedirs('models/gmms_conditional', exist_ok=True)
    best_model_path='ex3/models/gmms_conditional/gmms_INN.pt'
    min_losses = init_and_train_from_grid(hparams_grid, fixed_params, best_model_path, 'gmm', conditional = True)

    # Save results
    min_losses.to_csv('ex3/min_losses_conditional_gmm.csv', index=False)

