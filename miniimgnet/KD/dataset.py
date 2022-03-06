import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
import numpy as np
from skimage import io
import KD_tg as tg
import torchvision.transforms as transforms

class MiniImgnet(Dataset):
    def __init__(self, type, transform=None):
        self.transform = transform
        metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

        if type == 'train':
            folders = metatrain_folders
        else:
            folders = metatest_folders
        labels = np.array(range(len(folders)))
        labels = dict(zip(folders, labels))
        self.train_roots = []
        for c in folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            self.train_roots.extend(temp)
        random.shuffle(self.train_roots)

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def __getitem__(self, index):
        x_path = os.path.abspath('./' + self.train_roots[index])
        x = io.imread(x_path)
        if self.transform:
            x = self.transform(x)
        y = int(self.train_labels[index])
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.train_roots)

    def debug(self):
        print(self.train_roots[0])

def get_loader(type):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform=transforms.Compose([transforms.ToTensor(),normalize,transforms.Resize(224)])
    dataset = MiniImgnet(type, transform=transform)
    loader = DataLoader(dataset, batch_size=32)
    return loader

def get_loader_sm(type):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform=transforms.Compose([transforms.ToTensor(),normalize])
    dataset = MiniImgnet(type, transform=transform)
    loader = DataLoader(dataset, batch_size=32)
    return loader

# loader = MiniImgnet('test')
# loader.debug()