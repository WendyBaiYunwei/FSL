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
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        # x = x.view(x.size(0), -1)
        return x

def main():
    teacher = CNN()
    teacher.load_state_dict(torch.load('./base_teacher.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    student = CNN()
    student.load_state_dict(torch.load('./student.pth'))
    for param in student.parameters():
        param.requires_grad = False

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(),
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=1, 
                                        shuffle=True, 
                                        num_workers=1)

    for inputs, _ in trainloader:
        for channel_i in range(1, 4):
            sample_features = student(Variable(inputs)).view((7, 7, 32)).detach()[:, :, channel_i].squeeze()
            baseline_features = teacher(Variable(inputs)).view((7, 7, 32)).detach()[:, :, channel_i].squeeze()
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
        break

    # to-do: visualize feature vectors
if __name__ == '__main__':
    main()
