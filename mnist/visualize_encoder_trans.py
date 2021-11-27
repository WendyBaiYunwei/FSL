import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
from self_attention_cv import TransformerEncoder
import numpy as np
import os
import argparse
import math
from my_fc import MyLinearLayer
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

EPOCH = 10
BATCH_SIZE = 1
DIM = 10
tokenSize = 5
cropIs = [tokenSize * i for i in range(1, DIM // tokenSize + 1)]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,           
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 8, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Linear(10 * 10, 14),                                                   
        )
        self.conv2 = nn.Sequential(         
            nn.Linear(14, 10 * 10),                                        
        )
        self.out = nn.Linear(10 * 10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def getCrops(inputs):
    inputs = inputs.squeeze(dim = 1)#btch, channel, h, w
    batch = np.zeros((BATCH_SIZE, (DIM ** 2) // (tokenSize ** 2), tokenSize, tokenSize))
    for batchI, input in enumerate(inputs):
        tokenI = 0
        for i in cropIs:
            for j in cropIs:
                token = input[i - tokenSize:i, j - tokenSize:j]
                batch[batchI, tokenI, :, :] = token
                tokenI += 1
    batch = torch.from_numpy(batch)
    batch = torch.flatten(batch, start_dim = -2)
    return batch

def main():
    teacher = TransformerEncoder(dim=tokenSize ** 2,blocks=3,heads=2)
    teacher.load_state_dict(torch.load('./base_trans2.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    student = Encoder()
    student.load_state_dict(torch.load('./trans_student.pth'))
    for param in student.parameters():
        param.requires_grad = False

    transform = transforms.Compose(
            [transforms.Resize((10, 10)),
                transforms.ToTensor(),
            ])

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transform,
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        num_workers=1)
    i = 0
    for inputs, _ in trainloader:
        inputs_sm = inputs.flatten(start_dim = 1)
        sample_features = student(Variable(inputs_sm)).view((10, 10)).detach().squeeze()
        inputs = getCrops(inputs)
        baseline_features = teacher(Variable(inputs).float()).view((10, 10)).detach().squeeze()
        img = inputs[0].detach()
        arr = [sample_features, img.squeeze(), baseline_features]
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
        i+=1
        if i == 6:
            break

    # to-do: visualize feature vectors
if __name__ == '__main__':
    main()
