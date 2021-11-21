import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
import argparse
import math
import logging
from datetime import datetime
import argparse
from copy import deepcopy


## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()

EPISODE = 14
LEARNING_RATE = args.learning_rate

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
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x

def weights_init(m):
    classname = m.__class__.__name__
    teacher = CNN()
    teacher.load_state_dict(torch.load('./base_teacher.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    if classname.find('Linear') != -1:
        m.weight.data = deepcopy(teacher.out.weight.data)
        m.bias.data = deepcopy(teacher.out.bias.data)


def main():
    device = torch.device("cuda")
    
    student = CNN()
    student.load_state_dict(torch.load('./student.pth'))
    for param in student.parameters():
        param.requires_grad = False
    student.apply(weights_init).to(device)

    test_data = datasets.MNIST(
        root = 'data',
        train = False,                         
        transform = ToTensor(),
        download = False,            
    )

    testloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=1, 
                                        shuffle=True, 
                                        num_workers=1)
    print("Testing...")
    
    student.eval()
    
    def test():
        with torch.no_grad():
            for images, labels in testloader:
                test_output, _ = student(Variable(images).to(device))
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                labels = Variable(labels).to(device)
                accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        print('Test Accuracy of the model on the 10000 test images: %.5f' % accuracy)
    test()

if __name__ == '__main__':
    main()