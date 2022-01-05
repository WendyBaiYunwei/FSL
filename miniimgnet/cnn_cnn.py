import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
from tinyDs import TinyImgNetDs
import argparse
import math
import numpy as np
from torchvision import datasets, models
import os
from copy import deepcopy

parser = argparse.ArgumentParser()
args = parser.parse_args()

torch.manual_seed(0)

EPOCH = 2
BATCH_SIZE = 1
DIM = 84
DIM2 = 6
HIDDEN = False
studentPth = './cnn_student.pth'
teacherPth = './vgg16.pth'
lFunc = nn.CrossEntropyLoss()

class CNNstudent(nn.Module):
    def __init__(self):
        super(CNNstudent, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
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
            nn.Conv2d(32, 64, 3, 1, 1),     
            nn.ReLU(),
        )
        self.inter1 = nn.Linear(19 * 19 * 64, 5000)
        self.inter2 = nn.Linear(5000, 7 * 7 * 512)
        self.out = nn.Linear(7 * 7 * 512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim = 1)
        x = self.inter1(x)
        x = self.inter2(x)
        y = self.out(x)
        return x, y

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
    loss = torch.square(target.flatten(start_dim = 1) - out)
    return loss

def train(trainloader, trainloader_lg, feature, student, optimizer, scheduler, device):
    print("Training...")
    student.train()

    for i in range(EPOCH):
        epoch_loss = 0
        count = 0
        dataiter_lg = iter(trainloader_lg)
        for inputs, _ in trainloader:
            sample_features, _ = student(Variable(inputs).to(device))

            inputs_lg, _ = next(dataiter_lg)
            baseline_features = feature(Variable(inputs_lg).to(device)) # 16 * 32 * 7 * 7

            optimizer.zero_grad()

            loss = get_loss(sample_features, baseline_features)

            loss.backward(torch.ones_like(sample_features))

            optimizer.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1
            # scheduler.step()
        torch.save(student.state_dict(), studentPth)

def trainClassifier(trainloader, student, optimizer, device):
    student.train()
    count = 0
    for inputs, label in trainloader:
        if count % 10000 == 0:
            print(count)
        count += 1
        x, y = student(Variable(inputs).to(device))
        optimizer.zero_grad()

        label = Variable(label).to(device)
        loss = lFunc(y, label)
        loss.backward()

        optimizer.step()

def test(testloader, model, device):
    print("Testing...")
    model.eval()
    accuracy = 0
    count = 0
    for inputs, labels in testloader:
        _, output = model(Variable(inputs).to(device))
        pred_y = torch.max(output, 1)[1].data.squeeze()
        labels = Variable(labels).to(device)
        accuracy += (pred_y == labels).sum().item()
        count += 1
        if count % 1000 == 0:
            print(count)
    print('Test Accuracy of the model on the 10000 test images:', accuracy  / 10000 * 100)
    return accuracy

def main():
    device = torch.device("cuda")

    # assert os.path.exists(teacherPth)
    teacher = models.vgg16(pretrained=False)
    teacher.load_state_dict(torch.load('./vgg16.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.classifier[6] = nn.Sequential(nn.Linear(4096, 10)) ####
    feature = teacher.features
    classifier = teacher.classifier

    feature.to(device)
    classifier.to(device)
    
    student = CNNstudent()
    # student.load_state_dict(torch.load('./student_entry2.pth'))
    student.apply(weights_init)
    student.to(device)
    
    # optimizer = torch.optim.Adam([
    #     #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
    #     {"params": student.conv1.parameters(), "lr": 1e-3},
    #     {"params": student.conv2.parameters(), "lr": 1e-3},
    # ])
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer,step_size=10000,gamma=1.25)
    transform = transforms.Compose(
            [   transforms.ToTensor(),
                transforms.Resize(76),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    transform2 = transforms.Compose(
            [   transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    train_data = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    train_data_lg = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform2)
    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)
    trainloader_lg = torch.utils.data.DataLoader(train_data_lg, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)

    train(trainloader, trainloader_lg, feature, student, optimizer, scheduler, device)
    torch.save(student.state_dict(), studentPth)


    testloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=32, 
                                        shuffle=True, 
                                        num_workers=1)
    
    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": student.out.parameters(), "lr": 1e-3},
    ])

    trainClassifier(trainloader, student, optimizer, device) ##try freezing encoder
    test(testloader, student, device)

    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": student.out.parameters(), "lr": 1e-5},
    ])
    for i in range(5):
        trainClassifier(trainloader, student, optimizer, device) ##try freezing encoder
        test(testloader, student,  device)
        torch.save(student.state_dict(), studentPth)
    print('Done.')

if __name__ == '__main__':
    main()