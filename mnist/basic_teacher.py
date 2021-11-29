import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision import datasets, models
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
import math
import argparse
from copy import deepcopy
import torchvision.transforms as transforms

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
parser.add_argument("-hidden","--hidden",type = bool, default=True)
args = parser.parse_args()

LEARNING_RATE = args.learning_rate
HIDDEN = args.hidden

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Linear(13 * 13, 7),                                                   
        )
        self.conv2 = nn.Sequential(         
            nn.Linear(7, 8 * 7 * 7),                                        
        )
        self.out = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


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
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")

transform = transforms.Compose(
            [transforms.Resize((13, 13)),
                transforms.ToTensor(),
            ])
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = transform
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transform
)
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                        batch_size=100, 
                                        shuffle=False, 
                                        num_workers=1),
}

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

def train(num_epochs, cnn, loaders, optimizer):
        cnn.train()
            
        # Train the model
        total_step = len(loaders['train'])
            
        for epoch in range(num_epochs):####
            for i, (images, labels) in enumerate(loaders['train']):
                # images = images.flatten(start_dim = 1)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images).to(device)   # batch x
                b_y = Variable(labels).to(device)   # batch y
                output = cnn(b_x)            
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()    
                # apply gradients             
                optimizer.step()                
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test(student):
    # Test the model
    accuracy = 0
    student.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            # images = images.flatten(start_dim = 1)
            test_output = student(Variable(images).to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy += (pred_y == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: %.5f', accuracy / 10000)

def main():
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    train(15, cnn.to(device), loaders, optimizer)
    test(cnn)
    torch.save(cnn.state_dict(), './base_teacher.pth')

    # cnn.load_state_dict(torch.load('./base_teacher.pth'))
    # for param in cnn.parameters():
    #     param.requires_grad = False

    # student = Encoder()
    # student.load_state_dict(torch.load('./student_noactivate.pth'))
    # for param in student.parameters():
    #     param.requires_grad = False
    # student.out = nn.Linear(8 * 7 * 7, 10)
    # optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    # student.apply(weights_init)
    # student.to(device)
    # train(1, student, loaders, optimizer)
    # test(student)
    print('Done.')

if __name__ == '__main__':
    main()