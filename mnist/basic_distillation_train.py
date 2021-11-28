import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
import argparse
import math
import argparse


## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()

EPOCH = 3
BATCH_SIZE = 16
LEARNING_RATE = args.learning_rate
torch.manual_seed(0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=3,           
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 8, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.hidden = nn.Linear(8 * 7 * 7, 4)
        self.out = nn.Linear(4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
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
    teacher.load_state_dict(torch.load('./base_teacher_few_chnl.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)
    student = Encoder()
    student.apply(weights_init).to(device)
    student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer,step_size=1,gamma=0.9)

    transform = transforms.Compose(
                [#transforms.Resize((13, 13)),
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
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)
    trainloader_sm = torch.utils.data.DataLoader(train_data_sm, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)

    print("Training...")
    
    student.train()

    def train(EPOCH):
        epoch_loss = 0
        count = 0
        dataiter_sm = iter(trainloader_sm)
        for inputs, _ in trainloader:
            baseline_features = teacher(Variable(inputs).to(device)).flatten(start_dim = 1) # 16 * 32 * 7 * 7
            inputs_sm, _ = next(dataiter_sm)
            inputs_sm = torch.flatten(inputs_sm, start_dim = 1)
            sample_features = student(Variable(inputs_sm).to(device))

            optimizer.zero_grad()

            loss = get_loss(sample_features, baseline_features)

            loss.backward(torch.ones_like(sample_features))

            optimizer.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1

    for EPOCH in range(EPOCH):
        train(EPOCH)
        scheduler.step()
        torch.save(student.state_dict(), './student_noactivate.pth')
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()