import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from skimage import io, transform


class TrashDataset(Dataset):
    """Trash dataset."""

    def __init__(self, csvFile, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csvFile, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0]+'.png')
        image = io.imread(img_name)
        weight = int(self.data.iloc[idx, 1])
        currClass = int(self.data.iloc[idx, 2])
        #weight = np.array([weight], dtype=float)#.reshape(-1, 2)
        sample = {'image': image, 'weight': weight, 'class': currClass}

        if self.transform:
            sample = self.transform(sample)

        return sample