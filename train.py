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
    writer = SummaryWriter(pjoin('runs', 'exp_1'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elastic_model = seam_model()
    x_indices = { 'training_indices': np.arange(3, 1502, 90),
                  'validation_indices': np.array([6, 9, 12, 15])
    }
    # Set up the dataloader for training dataset
    train_dataset = SeismicLoader2D(x_indices, elastic_model, mode='train')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=len(train_dataset),
                              shuffle=False)

    val_dataset = SeismicLoader2D(x_indices, elastic_model, mode='val')
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=len(val_dataset),
                              shuffle=False)

    x_indices = {'training_indices':[9],
                 'validation_indices':np.arange(3, 1495, 3)
    }
    test_dataset = SeismicLoader2D(x_indices, elastic_model, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


    # import tcn
    #model = Model().to(device)
    model = TCN(30,1,[3, 5, 5, 10, 20, 30, 40], 5, 0.4).to(device)

    for param in model.synthesis.parameters():
        param.requires_grad=True

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
        for x_train, y_train in train_loader:
            for x_test, y_test in test_loader:
                model.train()
                optimizer.zero_grad()
                y_pred, x_hat = model(x_train)
                _, x_hat = model(x_test)
                loss1 = criterion(y_pred, y_train)
                loss2 = criterion(x_hat, x_test)
                loss = loss1 + 0.5*loss2
                loss.backward()
                optimizer.step()
                train_loss.append(loss1.item())
                writer.add_scalar(tag='Training Loss', scalar_value=loss.item(), global_step=iter)
                if epoch % 20 == 0:
                    with torch.no_grad():
                        model.eval()
                        y_pred, x_hat = model(x_test)
                        loss = criterion(y_pred, y_test)
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

    # Set up directory to save results
    results_directory = 'results'
    with torch.no_grad():
        model.eval()
        for x,y in test_loader:
            AI_inv, seismic_reconstructed = model(x)
            AI_inv = AI_inv.detach().cpu()
            y = y.detach().cpu()
            titles=['Predicted Vp vs True Vp', 'Predicted Vs vs True Vs', 'Predicted Density vs True Density']
            for i in range(3):
                fig, (ax1,ax2) = plt.subplots(1,2)
                ax1.imshow(AI_inv[:,i,:].transpose(0,1))
                ax2.imshow(y[:,i,:].transpose(0,1))
                plt.suptitle(titles[i])
                plt.show()
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

