from torch.utils.data import Dataset
import torch
import segyio
from os.path import join as pjoin
import numpy as np
import sys

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
class SeismicLoader2D(Dataset):

    def __init__(self, x_indices, model, mode):
        """x_indices specify the indices of the training and validation well-logs in the model
           model is 2-D model of shape xline x depth
        """

        # Normalize and standardize the training data
        self.seismic = segyio.cube(pjoin('data', 'SEAM_Interpretation_Challenge_1_Depth.sgy'))
        self.x_indices = x_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.section, self.model, self.indices = self.standardize(mode)

    def standardize(self, mode):
        seismic = self.seismic
        seismic_normalized = (seismic - seismic.mean()) / seismic.std()
        section = torch.tensor(seismic_normalized[714], dtype=torch.float).to(self.device)  # get the matching seismic section at 23910 m compared to model at 23900 m
        model = self.model
        train_indices = self.x_indices['training_indices']
        model = torch.tensor((model - model[train_indices].mean()) / model[train_indices].std(),
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
        seismic_index = np.int(model_index/3 * 2)
        x = self.section[seismic_index-1:seismic_index+2]  # each seismic input is 3 traces thick and full depth in length
        x = torch.transpose(x,0,1)  # arrange it in H x W format
        x = torch.unsqueeze(x, 0).to(self.device)  # Add channel dimension at the 0th place
        y = self.model[model_index].to(self.device)
        return x, y

    def __len__(self):
        return len(self.indices)  # All the wells are usable except the 1st and the last, since they have no seismic traces to their left and right


# Define a class for invalid indices
class InvalidIndex(Exception):
    """Raised when an invalid index is used for the model"""
    def __init__(self):
        pass


