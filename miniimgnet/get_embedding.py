# get embedding
from skimage import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import task_generator as tg
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
# input training dataset
# output embedding for each training sample
# save as pickle files, N * feature dimension

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

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def __getitem__(self, index):
        x_path = os.path.abspath('./' + self.train_roots[index])
        x = io.imread(x_path)
        x = self.transform(x)
        y = int(self.train_labels[index])
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.train_roots)

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64


normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
transform=transforms.Compose([transforms.ToTensor(),normalize])
dataset = MiniImgnet(transform)
data = DataLoader(dataset)

# load weights
enc = CNNEncoder()
# enc.cuda()
enc.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(5) +"way_" + str(1) +"shot.pkl")))

yList = []
xList = [] 
k = 0
for x, y in data:
    k += 1

    if k % 1000 == 0 or k == 38400:
        print(k)
        xs = torch.stack(xList)
        ys = torch.stack(yList)
        with open('./train_x/' + str(k) + '.pkl', 'wb') as out:
            pickle.dump(xs, out)

        with open('./train_y/' + str(k) + '.pkl', 'wb') as out:
            pickle.dump(ys, out)
        
        del yList[:]
        del xList[:]
        yList = []
        xList = [] 

    # x = x.cuda()
    x = enc(x)
    xList.append(x)
    yList.append(y)