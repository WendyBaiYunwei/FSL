import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import math
from my_fc import MyLinearLayer
from cifar_generator import CIFAR10 
import matplotlib.pyplot as plt

EPISODE = 0
LEARNING_RATE = 0.1

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer6 = MyLinearLayer(7, 7, 512)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

def main():
    feature_encoder = CNNEncoder()
    assert os.path.exists('./checkpoint/vgg16.pth')
    model = models.vgg16(pretrained=False)
    model.load_state_dict(torch.load('./checkpoint/vgg16.pth'))
    vgg16 = model.features

    assert os.path.exists('./feature_encoder.pth')
    feature_encoder.load_state_dict(torch.load('./feature_encoder.pth'))

    # Freeze model parameters
    for param in vgg16.parameters():
        param.requires_grad = False
    
    transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    trainset = CIFAR10(root='../datas/cifar-10-python', train=True,
                                    download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    i = 0
    for inputs, _ in trainloader:
        sample_features = feature_encoder(Variable(inputs)).view((7, 7, 512)).detach()[:, :, 0]
        baseline_features = vgg16(Variable(inputs)).detach()[0, 0, :, :]
        img = inputs[0].view(224, 224, 3).detach()
        arr = [sample_features, img, baseline_features]
        ix = 1
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(1, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(arr[j], cmap='gray')
            ix += 1
        plt.show()
        break

    # to-do: visualize feature vectors
if __name__ == '__main__':
    main()
