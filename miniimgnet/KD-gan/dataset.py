import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
import numpy as np
from skimage import io
import task_generator as tg
import torchvision.transforms as transforms

class MiniImgnet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
        labels = np.array(range(len(metatrain_folders)))
        labels = dict(zip(metatrain_folders, labels))
        self.train_roots = []
        for c in metatrain_folders:
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

def get_loader():
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform=transforms.Compose([transforms.ToTensor(),normalize,transforms.Resize(224)])
    dataset = MiniImgnet(transform=transform)
    loader = DataLoader(dataset, batch_size=32)
    return loader

def get_loader_sm():
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform=transforms.Compose([transforms.ToTensor(),normalize])
    dataset = MiniImgnet(transform=transform)
    loader = DataLoader(dataset, batch_size=32)
    return loader