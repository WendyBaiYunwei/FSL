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
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

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
            nn.Linear(14 * 14, 7),                              
            nn.ReLU(),                       
        )
        self.conv2 = nn.Sequential(         
            nn.Linear(7, 8 * 7 * 7),                                        
        )
        self.out = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def main():
    teacher = CNN()
    teacher.load_state_dict(torch.load('./base_teacher_few_chnl.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    student = Encoder()
    student.load_state_dict(torch.load('./student.pth'))
    for param in student.parameters():
        param.requires_grad = False

    transform = transforms.Compose(
            [transforms.Resize((14, 14)),
                transforms.ToTensor(),
            ])
    train_data_sm = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transform,
        download = False,            
    )

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transforms.ToTensor(),
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        num_workers=1)
    trainloader_sm = torch.utils.data.DataLoader(train_data_sm, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        num_workers=1)
    i = 0
    dataiter_sm = iter(trainloader_sm)
    for inputs, _ in trainloader:
        inputs_sm, _ = next(dataiter_sm)
        inputs_sm = inputs_sm.flatten(start_dim = 1)
        for channel_i in range(1, 4):
            sample_features = student(Variable(inputs_sm)).view((7, 7, 8)).detach()[:, :, channel_i].squeeze()
            baseline_features = teacher(Variable(inputs)).view((7, 7, 8)).detach()[:, :, channel_i].squeeze()
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
