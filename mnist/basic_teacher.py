import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision import datasets, models
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
import argparse
import argparse


## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()

EPISODE = 10
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
        return output

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")
cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor()            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1),
}

def train(num_epochs, cnn, loaders):
        cnn.train()
            
        # Train the model
        total_step = len(loaders['train'])
            
        for epoch in range(EPISODE):
            for i, (images, labels) in enumerate(loaders['train']):
                    
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

def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            test_output = cnn(Variable(images).to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

def main():
    # train(EPISODE, cnn, loaders)
    # torch.save(cnn.state_dict(), './base_teacher.pth')
    cnn.load_state_dict(torch.load('./base_teacher.pth'))
    for param in cnn.parameters():
        param.requires_grad = False
    test()
    print('Done.')

if __name__ == '__main__':
    main()