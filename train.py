# This script trains the model defined in model file on the marmousi post-stack seismic gathers
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from core.utils import *
from core.data_loader import *
from core.model import *
from core.results import *

# Fix the random seeds
#torch.manual_seed(2561716316833428258)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


# Define function to perform train-val split
def train_val_split(args):
    """Splits dataset into training and validation based on the number of well-logs specified by the user.

    The training traces are sampled uniformly along the length of the model. The validation data is all of the
    AI model except the training traces. Mean and Standard deviation are computed on the training data and used to
    standardize both the training and validation datasets.
    """
    # Load data
    seismic = seam_seismic().squeeze()  # dim= No_of_gathers x trace_length
    impedance = seam_model()  # dim = No_of_traces x trace_length


    # Split into train and val
    train_indices = np.linspace(2, len(seismic)-3, args.n_wells).astype(int)
    val_indices = np.setdiff1d(np.arange(1, len(seismic)-2).astype(int), train_indices)
    x_train = np.expand_dims(np.array([seismic[i-2:i+3] for i in train_indices]), axis=1).transpose(0, 1, 3, 2)
    y_train = impedance[train_indices].reshape(len(train_indices), 1, impedance.shape[1], 1)
    x_val = np.expand_dims(np.array([seismic[i-2:i+3] for i in val_indices]), axis=1).transpose(0,1,3,2)
    y_val = impedance[val_indices].reshape(len(val_indices), 1, impedance.shape[1], 1)
    seismic = np.expand_dims(np.array([seismic[i-2:i+3] for i in range(1, len(seismic)-2)]), axis=1).transpose(0,1,3,2)

    # Standardize features and targets
    x_train_norm, y_train_norm = (x_train - x_train.mean()) / x_train.std(), (y_train - y_train.mean()) / y_train.std()
    x_val_norm, y_val_norm = (x_val - x_train.mean()) / x_train.std(), (y_val - y_train.mean()) / y_train.std()
    seismic = (seismic - x_train.mean()) / x_train.std()

    return x_train_norm, y_train_norm, x_val_norm, y_val_norm, seismic


# Define train function
def train(args):
    """Sets up the model to train"""
    # Create a writer object to log events during training
    writer = SummaryWriter(pjoin('runs', 'exp_1'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AI = seam_model()
    x_indices = { 'training_indices': np.arange(3, 1502, 30),
                  'validation_indices': np.array([6, 9, 12, 15])
    }
    # Set up the dataloader for training dataset
    train_dataset = SeismicLoader2D(x_indices, AI, mode='train')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=len(train_dataset),
                              shuffle=False)

    val_dataset = SeismicLoader2D(x_indices, AI, mode='val')
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=len(val_dataset),
                              shuffle=False)
    # import tcn
    model = TCN(1,
                1,
                args.tcn_layer_channels,
                args.kernel_size,
                args.dropout).to(device)

    # Set up loss
    criterion = torch.nn.MSELoss()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.weight_decay,
                                 lr=args.lr)

    # Set up list to store the losses
    train_loss = [np.inf]
    val_loss = [np.inf]
    iter = 0
    # Start training
    for epoch in range(args.n_epoch):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar(tag='Training Loss', scalar_value=loss.item(), global_step=iter)
            if epoch % 200 == 0:
                with torch.no_grad():
                    model.eval()
                    for x,y in val_loader:
                        y_pred = model(x)
                        loss = criterion(y_pred, y)
                        val_loss.append(loss.item())
                        writer.add_scalar(tag='Validation Loss', scalar_value=loss.item(), global_step=iter)

            print('epoch:{} - Training loss: {:0.4f} | Validation loss: {:0.4f}'.format(epoch,
                                                                                        train_loss[-1],
                                                                                        val_loss[-1]))

            # if epoch % 100 == 0:
            #     with torch.no_grad():
            #         model.eval()
            #         AI_inv = model(seismic)
            #     fig, ax = plt.subplots()
            #     ax.imshow(AI_inv[:, 0].detach().cpu().numpy().squeeze().T, cmap="rainbow")
            #     ax.set_aspect(4)
            #     writer.add_figure('Inverted Acoustic Impedance', fig, iter)
        iter += 1

    writer.close()

    x_indices = {'training_indices':[9],
                 'validation_indices':np.arange(3, 1502,3)
    }
    test_dataset = SeismicLoader2D(x_indices, AI, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Set up directory to save results
    results_directory = 'results'
    with torch.no_grad():
        model.eval()
        for x,y in test_loader:
            AI_inv = model(x)


    if not os.path.exists(results_directory):  # Make results directory if it doesn't already exist
        os.mkdir(results_directory)
        print('Saving results...')
    else:
        print('Saving results...')

    np.save(pjoin(results_directory, 'AI.npy'), marmousi_model().T[452:2399, 400:2400])
    np.save(pjoin(results_directory, 'AI_inv.npy'), AI_inv.detach().cpu().numpy().squeeze())
    print('Results successfully saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs. Default = 1000')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                        help='Batch size. Default = 1.')
    parser.add_argument('--tcn_layer_channels', nargs='+', type=int, default=[3],
                        help='No of channels in each temporal block of the tcn. Default = numbers reported in paper')
    parser.add_argument('--kernel_size', nargs='?', type=int, default=5,
                        help='kernel size for the tcn. Default = 5')
    parser.add_argument('--dropout', nargs='?', type=float, default=0.2,
                        help='Dropout for the tcn. Default = 0.2')
    parser.add_argument('--n_wells', nargs='?', type=int, default=10,
                        help='# of well-logs used for training. Default = 19')
    parser.add_argument('--lr', nargs='?', type=float, default=0.001,
                        help='learning rate parameter for the adam optimizer. Default = 0.001')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=0.0001,
                        help='weight decay parameter for the adam optimizer. Default = 0.0001')

    args = parser.parse_args()
    train(args)
    evaluate(args)

