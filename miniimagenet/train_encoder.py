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
import os
import argparse
import math
from my_fc import MyLinearLayer
from cifar_generator import CIFAR10 

EPISODE = 1
LEARNING_RATE = 0.1

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        self.layer6 = MyLinearLayer(7, 7, 512)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_loss(out, target):
    # 7 * 7, 512 -> v * 512
    loss = torch.sum(torch.abs(out - target), 1)
    return torch.squeeze(loss)

def main():
    device = torch.device("cuda")
    
    feature_encoder = CNNEncoder()
    if os.path.exists('./checkpoint/vgg16.pth'):
        model = models.vgg16(pretrained=False)
        model.load_state_dict(torch.load('./checkpoint/vgg16.pth'))
        vgg16 = model.features
    else:
        vgg16 = models.vgg16(pretrained=True).features

    if os.path.exists('./checkpoint/feature_encoder.pth'):
        feature_encoder.load_state_dict(torch.load('./checkpoint/feature_encoder.pth'))
    else:
        feature_encoder.apply(weights_init)

    vgg16.to(device)
    feature_encoder.to(device)

    # Freeze model parameters
    for param in vgg16.parameters():
        param.requires_grad = False

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    
    transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    trainset = CIFAR10(root='./datas/cifar-10-python', train=True,
                                    download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    print("Training...")

    def train(episode):
        count = 0
        for inputs, _ in trainloader:

            sample_features = feature_encoder(Variable(inputs).to(device))

            baseline_features = vgg16(Variable(inputs).to(device)) # batch_size * 512 * 7 * 7

            baseline_features = torch.transpose(baseline_features, 1, 3)
            baseline_features = torch.flatten(baseline_features, end_dim=2)

            feature_encoder_optim.zero_grad()

            loss = get_loss(sample_features, baseline_features)
            loss.backward(torch.ones_like(loss))

            feature_encoder_optim.step()
            count += 1
            if count % 1000 == 0:
                print(count)

        feature_encoder_scheduler.step(episode)
        curr_loss = torch.sum(loss).item()
        print("episode:",episode+1)
        print("loss:",curr_loss)

    feature_encoder.train()

    for episode in range(EPISODE):
        train(episode)

    print("Done training, start saving")

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(feature_encoder.state_dict(), './checkpoint/feature_encoder.pth')

    print('Done.')

    # to-do: visualize feature vectors
if __name__ == '__main__':
    main()
