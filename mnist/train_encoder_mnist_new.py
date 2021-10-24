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
import logging
from datetime import datetime
import argparse

# resume encoder from, change vgg16 dir

parser = argparse.ArgumentParser()
parser.add_argument("-l1","--learning_rate",type = float, required=True) # estimate: 0.2
parser.add_argument("-l2","--learning_rate_fc",type = float, required=True) # estimate: 0.0001
parser.add_argument("-s","--step_size",type = int, default = 1000000)
parser.add_argument("-v","--version_number",type = str, required=True)
parser.add_argument("-v_from","--resumed_v",type = str, required=True)
args = parser.parse_args()

EPISODE = 1000
LEARNING_RATE = args.learning_rate
LEARNING_RATE_FC = args.learning_rate_fc
VERSION = args.version_number
VERSION2 = args.resumed_v
STEP_SIZE = args.step_size

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
    # 49, 512
    loss = torch.abs(out - target)
    return torch.squeeze(loss)

def main():
    logging.basicConfig(filename='record_mnist' + VERSION +'.log', level=logging.INFO)

    now = datetime.now()

    current_time = now.strftime("%m/%d-%H:%M:%S")
    logging.info(" Current Time = " + current_time)
    logging.info(VERSION + '-' + str(LEARNING_RATE) + '-' + str(LEARNING_RATE_FC) + '-' + str(STEP_SIZE))

    device = torch.device("cuda")
    
    if os.path.exists('./vgg16.pth'):
        model = models.vgg16(pretrained=False)
        model.load_state_dict(torch.load('./vgg16.pth'))
        vgg16 = model.features
    else:
        vgg16 = models.vgg16(pretrained=True).features

    feature_encoder = CNNEncoder()
    if VERSION2 != '0':
        feature_encoder.load_state_dict(torch.load('./feature_encoder_mnist' + VERSION2 + '.pth'))
    else:
        feature_encoder.apply(weights_init)

    vgg16.to(device)
    feature_encoder.to(device)

    # Freeze model parameters
    for param in vgg16.parameters():
        param.requires_grad = False

    feature_encoder_optim = torch.optim.Adam([
        {"params": feature_encoder.layer6.parameters(), "lr": LEARNING_RATE_FC},
        ], lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=STEP_SIZE,gamma=0.7)
    
    transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

    trainset = datasets.MNIST(root='./datas/mnist', download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=2)

    print("Training...")

    def train(episode):
        epoch_loss = 0
        count = 0
        for inputs, _ in trainloader:
            inputs = inputs.repeat(1, 3, 1, 1)
            sample_features = feature_encoder(Variable(inputs).to(device))

            baseline_features = vgg16(Variable(inputs).to(device)) # batch_size * 512 * 7 * 7

            baseline_features = torch.transpose(baseline_features, 1, 3)
            baseline_features = torch.flatten(baseline_features, end_dim=2)

            feature_encoder_optim.zero_grad()

            loss = get_loss(sample_features, baseline_features)
            loss.backward(torch.ones_like(loss))

            feature_encoder_optim.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 100 == 0:
                print(count, epoch_loss / (count + 1))
                feature_encoder_scheduler.step()
            count += 1

        now = datetime.now()
        current_time = now.strftime("%m/%d-%H:%M:%S")
        logging.info("episode:" + str(episode+1) + "  loss:" + str(epoch_loss / len(trainloader)) +\
            " learning_rate:" + str(feature_encoder_scheduler.get_last_lr()) + " time:" + str(current_time))
        

    feature_encoder.train()

    for episode in range(EPISODE):
        train(episode)
        torch.save(feature_encoder.state_dict(), './feature_encoder_mnist' + VERSION +'.pth')

    # print("Done training, start saving")

    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    # torch.save(feature_encoder.state_dict(), './checkpoint/feature_encoder_mnist.pth')

    print('Done.')

    # to-do: visualize feature vectors
if __name__ == '__main__':
    main()
