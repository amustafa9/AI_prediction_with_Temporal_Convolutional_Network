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


# Define train function
def train(args):
    """Sets up the model to train"""
    # Create a writer object to log events during training
    writer = SummaryWriter(pjoin('runs', 'exp_2'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_indices = { 'training_indices': np.arange(3, 1502, 114),
                  'validation_indices': np.arange(3,1502,90)
    }

    # Get project root directory
    root = os.getcwd()

    # Set up the dataloader for training dataset
    train_dataset_seam = SeamLoader2D(x_indices, root, mode='train', data='poststack')
    train_loader_seam = DataLoader(dataset=train_dataset_seam,
                              batch_size=len(train_dataset_seam),
                              shuffle=False)

    val_dataset_seam = SeamLoader2D(x_indices, root,  mode='val', data='poststack')
    val_loader_seam = DataLoader(val_dataset_seam, batch_size=len(val_dataset_seam))

    x_indices = {'training_indices': np.arange(3, 1502, 300),
                 'validation_indices': np.arange(3, 1496, 3)
                 }

    test_dataset_seam = SeamLoader2D(x_indices, root, mode='test', data='poststack')
    test_loader_seam = DataLoader(test_dataset_seam, batch_size=40)

    x_indices = {'training_indices': np.arange(451, 2199, 30),
                 'validation_indices': np.arange(451, 2199, 110)
                 }

    # Set up the dataloader for training dataset
    train_dataset_marmousi = MarmousiLoader2D(x_indices, mode='train', data='migrated')
    train_loader_marmousi = DataLoader(dataset=train_dataset_marmousi,
                              batch_size=len(train_dataset_marmousi),
                              shuffle=False)

    val_dataset_marmousi = MarmousiLoader2D(x_indices, mode='val', data='migrated')
    val_loader_marmousi = DataLoader(val_dataset_marmousi, batch_size=len(val_dataset_marmousi))

    x_indices = {'training_indices': np.arange(451, 2199, 30),
                 'validation_indices': np.arange(451, 2199, 1)
                 }

    test_dataset_marmousi = MarmousiLoader2D(x_indices, mode='test', data='migrated')
    test_loader_marmousi = DataLoader(test_dataset_marmousi, batch_size=40)

    # import tcn
    #model = Model().to(device)
    model = TCN(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)
    #model.load_state_dict(torch.load('models/best_val_model.pth'))

    # for param in model.synthesis.parameters():
    #     param.requires_grad = True

    # Set up loss
    criterion = torch.nn.MSELoss()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.weight_decay,
                                 lr=args.lr)

    # Set up list to store the losses
    iter = 0
    last_val_loss = np.inf
    # Start training
    for epoch in range(args.n_epoch):
        for x_train, y_train in train_loader_marmousi:
                model.train()
                optimizer.zero_grad()
                y_pred, x_hat = model(x_train)
                #_, x_hat = model(x_test)
                loss1 = criterion(y_pred, y_train)
                loss2 = criterion(x_hat, x_train)
                loss_marmousi = loss1 + 0.5*loss2
                writer.add_scalar('Marmousi_train_loss', loss_marmousi.item(), iter)

        for x_train, y_train in train_loader_seam:
                y_pred, x_hat = model(x_train)
                # _, x_hat = model(x_test)
                loss1 = criterion(y_pred, y_train)
                loss2 = criterion(x_hat, x_train)
                loss_seam = loss1 + 0.5 * loss2
                writer.add_scalar('Seam_train_loss', loss_seam.item(), iter)

        loss = loss_seam + loss_marmousi
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                model.eval()
                for x_val,y_val in val_loader_marmousi:
                    y_pred, x_hat = model(x_val)
                    val_loss_marmousi = criterion(y_pred, y_val)
                    writer.add_scalar('Marmousi_val_loss', val_loss_marmousi.item(), iter)
                for x_val,y_val in val_loader_seam:
                    y_pred, x_hat = model(x_val)
                    val_loss_seam = criterion(y_pred, y_val)
                    writer.add_scalar('SEAM_val_loss', val_loss_seam.item(), iter)
                    if val_loss_seam.item() <= last_val_loss:
                        torch.save(model.state_dict(), 'models/best_val_model.pth')
                        last_val_loss = val_loss_seam.item()
        print('epoch:{} - Training loss SEAM: {:0.4f} | Training loss Marmousi: {:0.4f} | Validation loss SEAM: {'
              ':0.4f} | Validation loss Marmousi: {:0.4f}'.format(epoch, loss_seam.item(), loss_marmousi.item(), val_loss_seam.item(), val_loss_marmousi.item()))

        iter += 1

    writer.close()

    # Set up directory to save results
    results_directory = 'results'
    l = len(test_dataset_seam)
    x,y = test_dataset_seam[0]
    Ip_inv = np.zeros((l,1,y.shape[-1]))
    Ip_actual = np.zeros((l, 1, y.shape[-1]))
    with torch.no_grad():
        model.eval()
        i = 0
        prev = 0
        for x, y in test_loader_seam:
            AI_inv, _ = model(x)
            l = len(AI_inv)
            Ip_inv[prev:len(AI_inv) + prev] = AI_inv.detach().cpu()
            Ip_actual[prev:len(AI_inv) + prev] = y.detach().cpu()
            i += 1
            prev = prev + l
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(Ip_inv.squeeze().T)
        ax2.imshow(Ip_actual.squeeze().T)
        plt.show()
        print('r2 Coefficient: {:0.4f}'.format(metrics(Ip_actual.squeeze().T, Ip_inv.squeeze().T)))
            # titles=['Predicted Vp vs True Vp', 'Predicted Vs vs True Vs', 'Predicted Density vs True Density']
            # for i in range(3):
            #     fig, (ax1,ax2) = plt.subplots(1,2)
            #     ax1.imshow(AI_inv[:,i,:].transpose(0,1))
            #     ax2.imshow(y[:,i,:].transpose(0,1))
            #     plt.suptitle(titles[i])
            #     plt.show()
    plt.imshow(AI_inv.detach().cpu().transpose(0, 1)), plt.xticks(ticks=np.linspace(0, AI_inv.shape[0], 5),
                                                                  labels=np.linspace(2490, 32520, 5))
    plt.axes().set_aspect(498 / 661), plt.colorbar()
    plt.xlabel('Distance Easting (Km)')
    plt.ylabel('Depth'), plt.show()

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
    parser.add_argument('--tcn_layer_channels', nargs='+', type=int, default=[3, 3, 3, 3, 3],
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

