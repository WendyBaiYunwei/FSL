import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import math

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()
torch.manual_seed(0)

LEARNING_RATE = args.learning_rate ####
EPOCH = 1
BATCH_SIZE = 1
DIM = 28
DIM2 = 6

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=4,            
                kernel_size=3,           
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(4, 4, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(4 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(len(x), -1)
        out = self.out(x)
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden1 = nn.Linear(DIM * DIM // 16 * DIM2, 4)
        self.classifier = nn.Linear(4, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden1(x)
        return x
    
def main():
    device = torch.device("cuda")
    teacher_cls = Classifier()
    teacher_cls.load_state_dict(torch.load('./base_classifier.pth'))
    for param in teacher_cls.parameters():
        param.requires_grad = False
    teacher_cls = teacher_cls.classifier
    teacher_cls.to(device)
    students = []
    for i in range(4):
        student = CNN()
        student.load_state_dict(torch.load('./trans_student_separate' + str(i) + '.pth'))
        for param in student.parameters():
            param.requires_grad = False
        student.to(device)
        students.append(student)
    # scheduler = StepLR(optimizer,step_size=1,gamma=0.98)

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ])
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transform,
        download = False,            
    )

    test_data = datasets.MNIST(
        root = 'data',
        train = False,                         
        transform = transform,
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1)
    testloader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1)


    # print("Training...")
    # loss_func = nn.CrossEntropyLoss()
    # def train():
    #     for inputs, label in trainloader:
    #         samples = []
    #         for i in range(4):
    #             student = students[i]
    #             sample = student(Variable(inputs).to(device)).squeeze().item()
    #             samples.append(sample)
    #         feature4 = torch.from_numpy(np.array(samples).reshape(len(inputs), -1))
    #         output = teacher_cls(Variable(feature4).to(device).float())
    #         loss = loss_func(output, label)##optimizer for classifier
    #         loss.backward()

    print("Testing...")
    
    def test():
        count = 0
        for inputs, labels in testloader:
            samples = []
            for i in range(4):
                student = students[i]
                sample = student(Variable(inputs).to(device)).squeeze().item()
                samples.append(sample)
            feature4 = torch.from_numpy(np.array(samples).reshape(len(inputs), -1))
            output = teacher_cls(Variable(feature4).to(device).float())
            pred_y = torch.max(output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            count += 1
            if count % 1000 == 0:
                print(count)
        print('Test Accuracy of the model on the 10000 test images: %.5f' % accuracy)

    test()
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()