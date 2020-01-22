from torch.utils.data import Dataset
import torch
import segyio
from os.path import join as pjoin
import numpy as np
import sys
from core.utils import *

# Set up the dataloader
class SeismicLoader(Dataset):
    def __init__(self, x_train, y_train):
        self.input = x_train
        self.target = y_train

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def __len__(self):
        return self.input.shape[0]


# Set up the 2-D seismic dataloader
class SeamLoader2D(Dataset):

    def __init__(self, x_indices, project_root, mode, data='prestack'):
        """x_indices specify the indices of the training and validation well-logs in the model
           model is 2-D model of shape xline x depth
        """

        # Check if valid arguments used
        valid_data = ['prestack', 'poststack']
        assert data in valid_data,"Please use a valid data type!"

        # Normalize and standardize the training data
        if data == 'prestack':
            self.seismic = np.load(os.path.join(project_root, 'data','prestack_seam_seismic.npy'))
        else:
            self.seismic = np.load(os.path.join(project_root, 'data','poststack_seam_seismic.npy'))
        self.x_indices = x_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = seam_model()
        self.mode = mode
        self.data = data
        self.section, self.model, self.indices = self.standardize(mode)

    def standardize(self, mode):
        #seismic = np.mean(self.seismic, axis=1, keepdims=True)  # Averaging all channels to get one channel
        seismic = self.seismic
        seismic_normalized = (seismic - seismic.mean(axis=(0,2), keepdims=True)) / seismic.std(axis=(0,2), keepdims=True)  # Normalizing channel to have 0 mean and 1 std
        #section = torch.tensor(seismic_normalized, dtype=torch.float).to(self.device)
        model = self.model

        # Comment the following two lines if you do not want water and salt region to bet cut down and model to downsampled
        section = seismic_normalized[:, :, 90:-100]  # Cut water from seismic
        model = model[:,:,180:-200][:,:,::2]  # Cut water from model and also downsample to bring to seismic resolution in depth

        train_indices = self.x_indices['training_indices']

        if self.data != 'prestack':
            model = model[:,[0],:] * model[:,[2],:]  # Calculate P-impedance if post-stack data used

        model = torch.tensor((model - model[train_indices].mean(axis=(0,2), keepdims=True)) / model[train_indices].std(axis=(0,2), keepdims=True),
                             dtype=torch.float).to(self.device)
        if mode == 'train':
            indices = self.x_indices['training_indices']
            try:
                if 0 in indices or len(self.model) - 1 in indices or all(index%3 != 0 for index in indices):
                    raise InvalidIndex
            except InvalidIndex:
                print('Error: One or more of the model trace indices is invalid.')
                sys.exit(1)
        else:
            indices = self.x_indices['validation_indices']
            try:
                if 0 in indices or len(self.model) - 1 in indices or all(index % 3 != 0 for index in indices):
                    raise InvalidIndex
            except InvalidIndex:
                print('Error: One or more of the model trace indices is invalid.')
                sys.exit(1)
        return section, model, indices

    def __getitem__(self, index):
        model_index = self.indices[index]
        seismic_index = np.int(model_index/3 * 2 + 1)
        x = self.section[seismic_index-3:seismic_index+4]  # each seismic input is 5 traces thick and full depth in length
        x = x.transpose(1, 2, 0) # arrange it in H x W format
        if np.random.uniform() >= 0.5:
            x = x[:,:,::-1]  # randomly flip 'x' horizontally
            x = np.copy(x)
        x = torch.tensor(x,dtype=torch.float).to(self.device)
        y = self.model[model_index].to(self.device)
        return x, y

    def __len__(self):
        return len(self.indices)  # All the wells are usable except the 1st and the last, since they have no seismic traces to their left and right


# Set up the 2-D seismic dataloader
class MarmousiLoader2D(Dataset):

    def __init__(self, x_indices, mode='train', data='migrated'):
        """x_indices specify the indices of the training and validation well-logs in the model
           model is 2-D model of shape xline x depth
        """

        # Normalize and standardize the training data
        if data=='migrated':
            self.seismic = segyio.cube(pjoin('data', 'Kirchhoff_PoSDM.segy'))
        else:
            self.seismic = segyio.cube(pjoin('data', 'SYNTHETIC.segy'))
        self.x_indices = x_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = segyio.cube(pjoin('data', 'MODEL_P-WAVE_VELOCITY_1.25m.segy')) * segyio.cube(pjoin('data', 'MODEL_DENSITY_1.25m.segy'))
        self.mode = mode
        self.data = data
        self.section, self.model, self.indices = self.standardize(mode)

    def standardize(self, mode):
        seismic = self.seismic.transpose(1,0,2)[:,:,100:]
        seismic_normalized = (seismic - seismic.mean(axis=(0,2), keepdims=True)) / seismic.std(axis=(0,2), keepdims=True)  # Normalizing channel to have 0 mean and 1 std
        section = torch.tensor(seismic_normalized, dtype=torch.float).to(self.device)
        model = self.model.transpose(1,0,2)[::5,:,::4][:,:,100:]

        train_indices = self.x_indices['training_indices']
        model = torch.tensor((model - model[train_indices].mean(axis=(0,2), keepdims=True)) /
                             model[train_indices].std(axis=(0,2), keepdims=True), dtype=torch.float).to(self.device)
        if mode == 'train':
            indices = self.x_indices['training_indices']
            try:
                if 0 in indices or len(self.model) - 1 in indices:
                    raise InvalidIndex
            except InvalidIndex:
                print('Error: One or more of the model trace indices is invalid.')
                sys.exit(1)
        else:
            indices = self.x_indices['validation_indices']
            try:
                if 0 in indices or len(self.model) - 1 in indices:
                    raise InvalidIndex
            except InvalidIndex:
                print('Error: One or more of the model trace indices is invalid.')
                sys.exit(1)
        return section, model, indices

    def __getitem__(self, index):
        model_index = self.indices[index]
        seismic_index = model_index
        x = self.section[seismic_index-3:seismic_index+4]  # each seismic input is 3 traces thick and full depth in length
        x = x.transpose(0,1).transpose(1,2)  # arrange it in H x W format
        y = self.model[model_index].to(self.device)
        return x, y

    def __len__(self):
        return len(self.indices)  # All the wells are usable except the 1st and the last, since they have no seismic traces to their left and right


# Define a class for invalid indices
class InvalidIndex(Exception):
    """Raised when an invalid index is used for the model"""
    def __init__(self):
        pass

