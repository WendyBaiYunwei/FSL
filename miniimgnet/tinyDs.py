# tiny imgnet dataset
from torch.utils.data import Dataset
import numpy as np
import  PIL
import os
import torch
from skimage import io, transform

class TinyImgNetDs(Dataset):
    def __init__(self, type, transform=None):
        xy = np.loadtxt(os.path.abspath('./TinyImageNet/'+type+'.txt'), dtype='S')
        self.type = type
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        self.transform = transform

    def __getitem__(self, index):
        x_path = os.path.abspath('TinyImageNet/' + self.x[index].decode('UTF-8'))
        x = io.imread(x_path)
        if self.transform:
            x = self.transform(x)
        y = int(self.y[index])
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)