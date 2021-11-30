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
from cifar_generator import CIFAR10

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
parser.add_argument("-hidden","--hidden",type = bool, default=True)
args = parser.parse_args()

LEARNING_RATE = args.learning_rate
HIDDEN = args.hidden
EPOCH = 10
DIM = 224

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")

transform = transforms.Compose(
            [transforms.Resize((DIM, DIM)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
train_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    transform = transform,
    download = False
)
test_data = datasets.CIFAR10(
    root = 'data', 
    train = False, 
    transform = transform,
    download = False
)
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=32, 
                                        shuffle=True, 
                                        num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
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

def train(epoch, num_epochs, teacher, loaders, optimizer):
        teacher.train()
            
        # Train the model
        total_step = len(loaders['train'])

        for i, (images, labels) in enumerate(loaders['train']):
            # images = images.flatten(start_dim = 1)
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y
            output = teacher(b_x)            
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
        for images, labels in loaders['test']:
            # images = images.flatten(start_dim = 1)
            test_output = model(Variable(images).to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy += (pred_y == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images:', accuracy / 10000 * 100)
    return accuracy

def main():
    teacher = models.vgg16()
    # teacher.load_state_dict(torch.load('./vgg16.pth'))
    # for params in teacher.parameters():
    #     params.requires_grad = False
    teacher.classifier[6] = nn.Sequential(nn.Linear(4096, 490), nn.Linear(490, 10))
    teacher.load_state_dict(torch.load('./vgg_teacher_test16.pth'))
    teacher.to(device)

    optimizer = torch.optim.SGD(teacher.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD([
    #     #{"params": student.hidden.parameters(), "lr": 0.001},####0.002
    #     {"params": teacher.features.parameters(), "lr": 0.001, "momentum": 0.9, "weight_decay":5e-4},
    #     {"params": teacher.classifier.parameters(), "lr": 0.003, "momentum": 0.9, "weight_decay":8e-4},
    # ])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    for i in range(EPOCH):
        train(i, EPOCH, teacher.to(device), loaders, optimizer)
        cur_acc = test(teacher)
        if cur_acc > best_acc:
            torch.save(teacher.state_dict(), './vgg_teacher_test16.pth')
            best_acc = max(cur_acc, best_acc)
        # scheduler.step()

    # teacher.load_state_dict(torch.load('./base_teacher.pth'))
    # for param in teacher.parameters():
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