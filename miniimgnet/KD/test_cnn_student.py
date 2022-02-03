import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision import datasets, models
import torchvision
import numpy as np
import math
import argparse
from copy import deepcopy
import torchvision.transforms as transforms
import torch.nn.functional as F

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()

LEARNING_RATE = args.learning_rate
EPOCH = 30
DIM = 76

class CNNEncoder(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        Args:
            params: (Params) contains num_channels
        """
        super(CNNEncoder, self).__init__()
        self.num_channels = 32
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4*4*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 10)       
        self.dropout_rate = 0.5

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(-1, 4*4*self.num_channels*4)             # batch_size x 4*4*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 10

        return s

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
train_data = torchvision.datasets.CIFAR10(
    root='./data-cifar10', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(
    root='./data-cifar10', train=False, download=True, transform=transform)
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=32, 
                                        shuffle=True, 
                                        num_workers=1),
    
    'val'  : torch.utils.data.DataLoader(test_data, 
                                        batch_size=32, 
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

def train(epoch, num_epochs, student, loaders, optimizer):
        student.train()
            
        # Train the model
        total_step = len(loaders['train'])

        for i, (images, labels) in enumerate(loaders['train']):
            # images = images.flatten(start_dim = 1)
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y
            output = student(b_x)            
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

def test(model):
    # Test the model
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['val']:
            # images = images.flatten(start_dim = 1)
            test_output = model(Variable(images).to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy += (pred_y == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images:', accuracy / 10000 * 100)
    return accuracy

def main():
    student = CNNEncoder()
    student.apply(weights_init)
    # student.load_state_dict(torch.load('./student_entry2.pth'))
    student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD([
    #     #{"params": student.hidden.parameters(), "lr": 0.001},
    #     {"params": student.features.parameters(), "lr": 0.001, "momentum": 0.9, "weight_decay":5e-4},
    #     {"params": student.classifier.parameters(), "lr": 0.003, "momentum": 0.9, "weight_decay":8e-4},
    # ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    for i in range(EPOCH):
        train(i, EPOCH, student, loaders, optimizer)
        cur_acc = test(student)
        if cur_acc > best_acc:
            torch.save(student.state_dict(), './student_entry2.pth')
            best_acc = max(cur_acc, best_acc)
        scheduler.step()

    # student.load_state_dict(torch.load('./base_student.pth'))
    # for param in student.parameters():
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