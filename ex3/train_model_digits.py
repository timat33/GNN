

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter # install ' pillow ' to get PIL
import matplotlib.pyplot as plt
# define a functor to downsample images
from train_model_conditional import *

class DownsampleTransform:
    def __init__(self, target_shape, algorithm = Image.Resampling.LANCZOS ) :
        self.width, self.height = target_shape
        self.algorithm = algorithm

    def __call__(self, img ) :
        img = img.resize((self.width +2, self.height +2), self.algorithm )
        img = img.crop((1, 1, self.width +1, self.height +1) )
        return img
    
# concatenate a few transforms
transform = transforms.Compose([
    DownsampleTransform(target_shape = (8,8) ),
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor()
])

def get_mnist():
    # download MNIST
    mnist_dataset = datasets.MNIST(root = './data', train = True, 
                                   transform = transform, download = True)

    return mnist_dataset

def train_model(model, n_epoch, loss_fn, train_loader, conditional, lr, model_path, batch_size = 4, seed = 11121, device = 'cpu'):
    # Get extra necessary objects
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Move to gpu if possible
    x_train = x_train.to(device)

    # Define counter for early stopping to avoid overfitting/computation inefficiency
    early_stop_counter = 0
    early_stop_counter_max = 15 # Stop if no improvement in val loss after this many epochs

    # Train. Terminate early based on validation loss
    best_train_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch_index in range(n_epoch):

        model.train()
        # Train and get validation loss
        train_loss = train_epoch(model, train_loader, optimiser, loss_fn, conditional)

        # print outputs
        if epoch_index % 10 == 0:
            print('EPOCH {}:'.format(epoch_index + 1))
            print(f'  training batch loss: {train_loss}')

        # Store losses
        history['train_loss'].append(train_loss)

        # If best loss is beat, then keep going. Else increment counter. Stop if counter gets too high
        if train_loss < best_train_loss:
            early_stop_counter = 0 
            best_train_loss = train_loss
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': best_train_loss,
                'history': history
            }, model_path)

        else:
            early_stop_counter +=1
            if early_stop_counter == early_stop_counter_max:
                print(f'Early stopping after {epoch_index + 1} epochs')
                break

    return history

if __name__ == '__main__':
    # Unconditional stuff
    # Hparams
    hparams = Namespace()
    fixed_params = Namespace()

    ## Architecture hparams
    hparams.hidden_size = 36
    hparams.blocks = 24

    ## Training hparams
    hparams.lr = 0.01
    hparams.n_epoch = 200

    # Fixed params
    fixed_params.input_size = 64
    fixed_params.batch_size = 16
    fixed_params.seed = 11121
    fixed_params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_params.conditional = False
    fixed_params.positive_labels = None
    fixed_params.condition_size = 0

    results = pd.DataFrame(columns = ['k', 'min_val_loss'])
    for k in [0,2,4,8,16]:
        model_path = f'ex3/models/digits/digits{k}.pt'
        os.makedirs('ex3/models/digits', exist_ok=True)
        history = init_and_train(hparams, fixed_params, model_path, 'digits', k)

        # Get minimum validation loss
        min_val_loss = min(history['val_loss'])
        
        # Add row to results DataFrame
        results.loc[len(results)] = [int(k), min_val_loss]
    
    # Conditional stuff

    # Hparams
    hparams_grid = Namespace()
    fixed_params = Namespace()

    ## Architecture hparams
    hparams_grid.hidden_size = 36
    hparams_grid.blocks = 24

    ## Training hparams
    hparams_grid.lr = 0.01
    hparams_grid.n_epoch = 3

    # Fixed params
    fixed_params.input_size = 2 
    fixed_params.condition_size = 1
    fixed_params.conditional = True
    fixed_params.batch_size = 16
    fixed_params.seed = 11121
    fixed_params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_params.positive_labels = None # For gmm dataset
