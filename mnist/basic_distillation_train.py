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
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_loss(out, target):
    # 49, 512
    loss = torch.square(out - target)
    # loss = torch.squeeze(loss)
    return loss

def main():
    device = torch.device("cuda")
    
    teacher = CNN()
    teacher.load_state_dict(torch.load('./base_teacher.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)
    student = CNN()
    student.apply(weights_init).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    
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
    print("Training...")
    
    student.train()
    
    def train(episode):
        epoch_loss = 0
        count = 0
        for inputs, _ in trainloader:
            _, sample_features = student(Variable(inputs).to(device))

            _, baseline_features = teacher(Variable(inputs).to(device)) # batch_size * 512 * 7 * 7

            optimizer.zero_grad()

            loss = get_loss(sample_features, baseline_features)

            loss.backward(torch.ones_like(sample_features))

            optimizer.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1

    for episode in range(EPISODE):
        train(episode)
        if episode % 2 == 0:
            torch.save(student.state_dict(), './student.pth')
    print('Done.')

if __name__ == '__main__':
    main()