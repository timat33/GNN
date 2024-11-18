import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime


def get_dimensions(layers:int, input_size:int, hidden_size:int, bottleneck_size:int) -> np.array:
    """
    Get an np.array of layer dimensions to construct a set of linear layers. 
    Start from input_size, increase linearly in dimension up to hidden_size, then back down to bottleneck_size
    (ie encoder), then proceed symmetrically for the decoder.
    """
    # Obtain encoder layer dimensions 
    num_increasing_layers = np.floor(layers/2).astype(int)
    num_decreasing_layers = layers - num_increasing_layers
    increasing_layer_dims = np.linspace(input_size, hidden_size, num_increasing_layers, dtype = int)
    decreasing_layer_dims = np.linspace(hidden_size, bottleneck_size, num_decreasing_layers, dtype = int)
    encoder_layer_dims = np.concatenate([increasing_layer_dims,decreasing_layer_dims])

    # obtain decoder layer dims symmetric to encoder and concatenate together for all layer dims
    decoder_layer_dims = encoder_layer_dims[::-1] # Reverse and then skip first element (don't duplicate bottleneck)

    return encoder_layer_dims, decoder_layer_dims

def create_mlp_from_dim_array(dims: np.array) -> nn.Module:
    """
    Given an array of dimensions, create an mlp that has linear layers followed by ReLus except in the last step
    """
    layers = []
    for l in range(len(dims) - 1):
            # Add a Linear layer with the given input and output size
            layers.append(nn.Linear(dims[l], dims[l + 1]))
            # Add a ReLU activation function after each linear layer except after bottleneck and end
            if l < len(dims) - 2:
                layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    return model

class AutoEncoder(nn.Module):
    def __init__(self, input_size, bottleneck_size, hidden_size, layers):
        """
        Pass hidden size either as an int (in which case ramp linearly from input size to
        hidden size and back down to bottleneck size) or as a list of dimensions for each layer.
        """
        super().__init__()

        # Store architecture parameters
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size
        self.layers = layers

        # Obtain layer dimensions if an int is passed for hidden_size
        if isinstance(hidden_size, int):
            encoder_dims, decoder_dims = get_dimensions(layers, input_size, hidden_size, bottleneck_size)
        else:
            encoder_dims = np.array(hidden_size)
            decoder_dims = np.array(hidden_size[::-1])


        # Make model of alternating linear layers and ReLu activations
        self.encoder = create_mlp_from_dim_array(encoder_dims)
        self.decoder = create_mlp_from_dim_array(decoder_dims)

    def forward(self, x):
        # Pass standardised data through model
        code = self.encoder(x)
        x_hat = self.decoder(code)

        return code, x_hat

def train_epoch(model, train_loader, epoch_index, tb_writer, optimiser, loss_fn, mmd_loss = False):
    running_loss = 0.
    last_loss = 0.
    metrics = {'MAE': 0} 

    for i, x_batch in enumerate(train_loader):
        # Zero out gradients
        optimiser.zero_grad()

        # Make predictions (discard codes)
        codes, x_hat = model(x_batch)

        # Get loss, metrics, and gradients
        if not mmd_loss:
            loss = loss_fn(x_hat, x_batch)
        else:
            loss = loss_fn(x_hat, codes, x_batch)
        loss.backward()
        metrics['MAE'] += nn.functional.l1_loss(x_hat, x_batch).detach() 

        # Update
        optimiser.step()

        # Write to tensorboard
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print(f'  batch {i+1} training MSE: {last_loss}')
            total_batches = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, total_batches)
            running_loss = 0.

    metrics['MAE'] = metrics['MAE']/len(train_loader) # Divide by number of batches to get average MAE
    print(f'  epoch {epoch_index+1} training MAE (batch averaged): {metrics["MAE"]}')

    tb_writer.flush() # Write batch losses for this epoch to disk
    return last_loss

def get_train_val_split(x_train, batch_size, seed = 11121):
    # Make a training/validation split: shuffle and then split 80/20
    if seed:
        torch.manual_seed(seed)

    shuffled_inds = torch.randperm(x_train.shape[0])
    x_train = x_train[shuffled_inds]
    cut_index = np.floor(0.8*x_train.shape[0]).astype(int)
    x_train, x_val = x_train[:cut_index], x_train[cut_index:]

    train_loader = DataLoader(x_train, batch_size = batch_size)
    val_loader = DataLoader(x_val, batch_size=4)

    return train_loader, val_loader

def get_val_loss(model, val_loader, loss_fn, mmd_loss = False):
    model.eval()
    val_loss = 0
    for x_batch in val_loader:
        code, x_hat = model(x_batch)

        if not mmd_loss:
            loss = loss_fn(x_hat, x_batch)
        else:
            loss = loss_fn(x_hat, code, x_batch)

        val_loss+=loss

    # Divide to get loss averaged over batches
    val_loss = val_loss/len(val_loader)

    return val_loss

def train_model(model, n_epoch, loss_fn, x_train, lr, mmd_loss = False, batch_size = 4, seed = 11121):
    # Get extra necessary objects
    tb_writer = SummaryWriter()
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Define counter for early stopping to avoid overfitting/computation inefficiency
    early_stop_counter = 0
    early_stop_counter_max = 5 # Stop if no improvement in val loss after this many epochs

    # Make a training/validation split: shuffle and then split
    train_loader, val_loader = get_train_val_split(x_train, batch_size, seed)

    # Train. Terminate early based on validation loss.
    best_val_loss = float('inf')

    for epoch_index in range(n_epoch):
        print('EPOCH {}:'.format(epoch_index + 1))

        model.train()
        _ = train_epoch(model, train_loader, epoch_index, tb_writer, optimiser, loss_fn, mmd_loss)

        # Get validation loss
        val_loss = get_val_loss(model, val_loader, loss_fn, mmd_loss)

        print(f'  validation batch loss: {val_loss}')
        total_batches = epoch_index * len(train_loader)
        tb_writer.add_scalar('Loss/validation', val_loss, total_batches)

        # If best loss is beat, then keep going. Else increment counter. Stop if counter gets too high
        if val_loss < best_val_loss:
            early_stop_counter = 0 # If best loss fals
            best_val_loss = val_loss
        else:
            early_stop_counter +=1
            if early_stop_counter == early_stop_counter_max:
                print(f'Early stopping after {epoch_index + 1} epochs')
                break

        
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
    loss_fn = nn.MSELoss()
    train_model(autoencoder, n_epoch, loss_fn, x_standardised, lr, mmd_loss = False)