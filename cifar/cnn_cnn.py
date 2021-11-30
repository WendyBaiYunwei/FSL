import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
from self_attention_cv import TransformerEncoder
import argparse
import math
import numpy as np
from torchvision import datasets, models
import os
from cifar_generator import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("-test","--isTest",type = bool, default=False)
args = parser.parse_args()

torch.manual_seed(0)

isTest = args.isTest
CHECKTEACHER = False
EPOCH = 1
BATCH_SIZE = 1
DIM = 28
DIM2 = 6
HIDDEN = False
studentPth = './cnn_student_freeze_encoder.pth'
teacherPth = './vgg_teacher_test16.pth'
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
            nn.Conv2d(32, 32, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),           
        )
        self.hidden = nn.Linear(7 * 7 * 32, 7 * 7 * 10)
        self.out = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim = 1)
        x = self.hidden(x)
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
    loss = torch.square(out - target)
    return loss

def train(trainloader, trainloader_sm, feature, classifier, hidden, student, optimizer, scheduler, device):
    print("Training...")
    student.train()

    for i in range(EPOCH):
        dataiter_sm = iter(trainloader_sm)
        epoch_loss = 0
        count = 0
        for inputs, _ in trainloader:
            inputs_sm, _ = next(dataiter_sm)
            sample_features, _ = student(Variable(inputs_sm).to(device))

            baseline_features = feature(Variable(inputs).to(device)) # 16 * 32 * 7 * 7
            baseline_features = classifier(baseline_features.flatten(start_dim = 1))
            baseline_features = hidden(baseline_features)

            optimizer.zero_grad()

            loss = get_loss(sample_features, baseline_features)

            loss.backward(torch.ones_like(sample_features))

            optimizer.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1
            scheduler.step()
        torch.save(student.state_dict(), studentPth)

def trainClassifier(trainloader, student, optimizer, device):
    student.train()

    for inputs, label in trainloader:
        _, y = student(Variable(inputs).to(device))
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

    assert os.path.exists(teacherPth)
    teacher = models.vgg16(pretrained=False)
    teacher.classifier[6] = nn.Sequential(nn.Linear(4096, 490), nn.Linear(490, 10))
    teacher.load_state_dict(torch.load(teacherPth))
    for param in teacher.parameters():
        param.requires_grad = False
    feature = teacher.features
    classifier = teacher.classifier[:6]
    hidden = teacher.classifier[6][0]

    feature.to(device)
    classifier.to(device)
    hidden.to(device)
    
    student = CNNstudent()
    student.apply(weights_init).to(device)

    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": student.conv1.parameters(), "lr": 0.004},
        {"params": student.conv2.parameters(), "lr": 0.004},
        {"params": student.hidden.parameters(), "lr": 0.01},
    ])

    scheduler = StepLR(optimizer,step_size=20000,gamma=0.97)
    train_data = CIFAR10(
        root = 'data',
        train = True,                         
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        download = False,            
    )

    train_data_sm = CIFAR10(
        root = 'data',
        train = True,                         
        transform = transforms.Compose([transforms.Resize((56, 56)), transforms.ToTensor()]),
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1)
    
    trainloader_sm = torch.utils.data.DataLoader(train_data_sm, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1)

    
    student.to(device)

    train(trainloader, trainloader_sm, feature, classifier, hidden, student, optimizer, scheduler, device)
    torch.save(student.state_dict(), studentPth)

    test_data = datasets.CIFAR10(
        root = 'data',
        train = False,                         
        transform = transforms.Compose([transforms.Resize((56, 56)), transforms.ToTensor()]),
        download = True,            
    )

    testloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1)

    trainloader = torch.utils.data.DataLoader(train_data_sm, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1)
    
    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": student.conv1.parameters(), "lr": 0},
        {"params": student.conv2.parameters(), "lr": 0},
        {"params": student.hidden.parameters(), "lr": 0},
        {"params": student.out.parameters(), "lr": 0.04},
    ])

    for i in range(3):
        trainClassifier(trainloader, student, optimizer, device) ##try freezing encoder
        test(testloader, student,  device)
    
    print('Done.')

if __name__ == '__main__':
    main()