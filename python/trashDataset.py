import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd


class TrashDataset(Dataset):
    """Trash dataset."""

    def __init__(self, csvFile, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csvFile = csvFile
        self.root_dir = root_dir
        self.transform = transform
        self.data = np.loadtxt(self.csvFile, delimiter=',', dtype=str)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pass
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        #print(self.data[idx][0])

        #img_name = os.path.join(self.root_dir, self.data[idx][0]+".png")
        #print(img_name)
        #image = np.load(img_name)
        #image = image.astype(np.float32)
        #image = image / 255.0
        #image = np.expand_dims(image, axis=0)
        #sample = {'image': image}

        #if self.transform:
        #    sample = self.transform(sample)

        #return sample