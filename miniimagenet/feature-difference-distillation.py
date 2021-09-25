#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import argparse
import random
from PIL import Image

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default = 2) #1000
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out # 64

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

def main():
    device = torch.device("cuda")
    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()

    feature_encoder.apply(weights_init)

    feature_encoder.to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)

    vgg16 = models.vgg16(pretrained=True).features

    # Freeze model parameters
    for param in vgg16.parameters():
        param.requires_grad = False

    vgg16.to(device)
    
    transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform2 = transforms.Compose(
                [transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_base = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=False, transform=transform2)
    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        degrees = random.choice([0,90,180,270])

        sample_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
        base_dataloader = torch.utils.data.DataLoader(trainset_base, batch_size=1, shuffle=False)

        # sample datas
        samples, _ = sample_dataloader.__iter__().next()
        base, _ = base_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples).to(device)) # 5(4)x64*5*5
        # sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,5,5)
        
        # baseline_features = vgg16(Variable(base).cuda(GPU))

        print("sample feature size:", sample_features.size())

        # plt.figure(figsize=(19, 19))
        # plt.imshow(sample_features[0, 0, :, :], cmap='gray')
        # print("baseline feature size:", baseline_features.size())
        
        im = Image.fromarray(sample_features[0][0].cpu().detach().numpy()) #512*19*19
        im_resized = im.resize((7, 7))
        sample_arr = np.asarray(im_resized)

        print(sample_arr.shape)
        break

        feature_encoder.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)

        feature_encoder_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data)

if __name__ == '__main__':
    main()
