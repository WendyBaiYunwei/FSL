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
tokenSize = 4
cropIs = [tokenSize * i for i in range(1, DIM // tokenSize + 1)]
studentPth = './cnn_cnn/cnn_student.pth'
teacherPth = './cnn_cnn/cnn_teacher.pth'
lFunc = nn.CrossEntropyLoss()

class CNNteacher(nn.Module):
    def __init__(self):
        super(CNNteacher, self).__init__()
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
        self.hidden = nn.Linear(8 * 7 * 7, 4 * 7 * 7) ##apply dropout
        self.out = nn.Linear(4 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        return x

class CNNstudent(nn.Module):
    def __init__(self):
        super(CNNstudent, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=4,            
                kernel_size=5,           
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(4, 4, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.classifier = nn.Linear(4 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        y = self.classifier(x)
        return x, y

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(4 * 7 * 7, 10)

    def forward(self, x):
        x = self.classifier(x)
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
    loss = torch.square(out - target)
    return loss

def train(trainloader, teacher, student, optimizer, scheduler, device):
    print("Training...")
    
    student.train()

    for i in range(EPOCH):
        epoch_loss = 0
        count = 0
        for inputs, _ in trainloader:
            sample_features, _ = student(Variable(inputs).to(device))

            baseline_features = teacher(Variable(inputs).to(device).float()) # 16 * 32 * 7 * 7

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
        output = model(Variable(inputs).to(device))
        _, pred_y = torch.max(output, 1)[1].data.squeeze()
        labels = Variable(labels).to(device)
        accuracy += (pred_y == labels).sum().item()
        count += 1
        if count % 1000 == 0:
            print(count)
    print('Test Accuracy of the model on the 10000 test images:', accuracy  / 10000 * 100)

def main():
    device = torch.device("cuda")

    teacher = CNNteacher()
    teacher.load_state_dict(torch.load(teacherPth))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)
    
    student = CNNstudent()
    student.apply(weights_init).to(device)
    optimizer = torch.optim.Adam([
    #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
    {"params": student.conv1.parameters(), "lr": 0.001},
    {"params": student.conv2.parameters(), "lr": 0.001},
    {"params": student.classifier.parameters(), "lr": 0.005},
    ])

    scheduler = StepLR(optimizer,step_size=10000,gamma=0.95)
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transforms.ToTensor(),
        download = True,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1)
    student.to(device)

    train(trainloader, teacher, student, optimizer, scheduler, device)

    test_data = datasets.MNIST(
        root = 'data',
        train = False,                         
        transform = transforms.ToTensor(),
        download = True,            
    )

    testloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1)

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1)
    bestAcc = 0
    for i in range(3):
        trainClassifier(trainloader, student, optimizer, device)
        acc = test(testloader, student,  device)
        if acc > bestAcc:
            torch.save(student.state_dict(), studentPth)
    
    print('Done.')

if __name__ == '__main__':
    main()