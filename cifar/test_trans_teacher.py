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
from self_attention_cv import TransformerEncoder, ResNet50ViT

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.001) # estimate: 0.2
parser.add_argument("-hidden","--hidden",type = bool, default=True)
args = parser.parse_args()

LEARNING_RATE = args.learning_rate
HIDDEN = args.hidden
EPOCH = 50
DIM = 32
BATCH_SIZE = 32

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(64 * 64, DIM * DIM)
        self.out = nn.Linear(DIM * DIM, 10)

    def forward(self, x):
        x = x.reshape(len(x), -1)
        x = self.hidden(x)
        x = self.out(x)
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

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")

transform = transforms.Compose(
            [#transforms.Resize((DIM, DIM)),
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

def train(epoch, num_epochs, teacher, loaders, optimizer, classifier):
        teacher.train()
            
        # Train the model
        total_step = len(loaders['train'])

        for i, (images, labels) in enumerate(loaders['train']):
            # images = images.flatten(start_dim = 1)
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y
            embedding = teacher(b_x.float())   
            output = classifier(embedding)           
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

def test(model, classifier):
    # Test the model
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            # images = images.flatten(start_dim = 1)
            embedding = model(Variable(images).to(device).float())
            test_output = classifier(embedding)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy += (pred_y == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images:', accuracy / 10000 * 100)
    return accuracy

def main():
    teacher = ResNet50ViT(img_dim=32, pretrained_resnet=True, 
                        blocks=6, classification=False, 
                        dim_linear_block=64, dim=64)
    teacher.to(device)

    classifier = Classifier()
    classifier.apply(weights_init)
    classifier.to(device)

    optimizer = torch.optim.Adam([
        #{"params": student.hidden.parameters(), "lr": 0.001}, ##train classifier
        {"params": teacher.parameters(), "lr": 0.001},
        {"params": classifier.hidden.parameters(), "lr": 0.005},
        {"params": classifier.out.parameters(), "lr": 0.0015},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

    best_acc = 0
    for i in range(EPOCH):
        train(i, EPOCH, teacher, loaders, optimizer, classifier)
        scheduler.step(i-1)
        cur_acc = test(teacher, classifier)
        if cur_acc > best_acc:
            torch.save(teacher.state_dict(), './trans_teacher_test.pth')
            best_acc = max(cur_acc, best_acc)
    print('Done.')

if __name__ == '__main__':
    main()