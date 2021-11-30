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
studentPth = './trans_learnt_student.pth'
teacherPth = './trans_teacher_test.pth'
lFunc = nn.CrossEntropyLoss()
tokenSize = 8
cropIs = [tokenSize * i for i in range(1, DIM // tokenSize + 1)]

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(12 * 192, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape(len(x), -1)
        x = self.hidden(x)
        x = self.out(x)
        return x

def getCrops(inputs):
    batch = np.zeros((len(inputs), (DIM ** 2) // (tokenSize ** 2), 3, tokenSize, tokenSize))
    for batchI, input in enumerate(inputs):
        tokenI = 0
        for i in cropIs:
            for j in cropIs:
                token = input[:, i - tokenSize:i, j - tokenSize:j]
                batch[batchI, tokenI, :, :, :] = token
                tokenI += 1
    batch = torch.from_numpy(batch)
    batch = torch.flatten(batch, start_dim = -3)
    return batch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_loss(out, target):
    loss = torch.square(out - target)
    return loss

def train(trainloader, student, teacher, optimizer, scheduler, device):
    print("Training...")
    student.train()

    for i in range(EPOCH):
        epoch_loss = 0
        count = 0
        for inputs, _ in trainloader:
            inputs = getCrops(inputs).float()
            sample_features = student(Variable(inputs).to(device))

            baseline_features = teacher(Variable(inputs).to(device)) # 16 * 32 * 7 * 7

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

def trainClassifier(trainloader, student, classifier, optimizer, device):
    student.train()

    count = 0
    for inputs, label in trainloader:
        count += 1
        if count % 100 == 0:
            print(count)
        inputs = getCrops(inputs).float()
        
        sample_features = student(Variable(inputs).to(device))

        # print(sample_features.shape)
        y = classifier(sample_features)
        optimizer.zero_grad()

        label = Variable(label).to(device)
        loss = lFunc(y, label)
        loss.backward()

        optimizer.step()

def test(testloader, model, classifier, device):
    print("Testing...")
    model.eval()
    accuracy = 0
    count = 0
    for inputs, labels in testloader:
        inputs = getCrops(inputs).float()
        sample_features = model(Variable(inputs).to(device))
        y = classifier(sample_features)
        pred_y = torch.max(y, 1)[1].data.squeeze()
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
    teacher = TransformerEncoder(dim=tokenSize ** 2 * 3,blocks=2,heads=8)
    for param in teacher.parameters():
        param.requires_grad = False

    teacher.to(device)

    student = TransformerEncoder(dim=tokenSize ** 2 * 3,blocks=6,heads=8)
    student.to(device)

    classifier = Classifier()
    classifier.apply(weights_init)
    classifier.to(device)

    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": student.parameters(), "lr": 0.00001},
    ])

    scheduler = StepLR(optimizer,step_size=10000,gamma=1.1)

    transform = transforms.Compose(
            [#transforms.Resize((DIM, DIM)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    train_data = CIFAR10(
        root = 'data',
        train = True,                         
        transform = transform,
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1)

    
    student.load_state_dict(torch.load(teacherPth))
    student.to(device)

    # train(trainloader, student, teacher, optimizer, scheduler, device)

    test_data = datasets.CIFAR10(
        root = 'data',
        train = False,                         
        transform = transforms.Compose([transforms.Resize((56, 56)), transforms.ToTensor()]),
        download = True,            
    )

    testloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=50, 
                                        shuffle=True, 
                                        num_workers=1)
    
    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": student.parameters(), "lr": 0.001},
        {"params": classifier.hidden.parameters(), "lr": 0.01},
        {"params": classifier.out.parameters(), "lr": 0.005},
    ])

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1)
    for i in range(3):
        trainClassifier(trainloader, student, classifier, optimizer, device) ##try freezing encoder
        test(testloader, student, classifier, device)
    
    print('Done.')

if __name__ == '__main__':
    main()