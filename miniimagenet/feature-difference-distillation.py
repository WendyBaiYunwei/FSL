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

parser = argparse.ArgumentParser(description="feature distillation")
parser.add_argument("-e","--episode",type = int, default = 2) #1000
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
args = parser.parse_args()


EPISODE = args.episode
LEARNING_RATE = args.learning_rate
FEATURE_DIM = 64
RELATION_DIM = 8

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=0),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer7 = nn.Linear(7*7, 7*7)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out

class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
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
    loss = torch.sum(torch.sum(torch.abs(out - target), 3), 2)
    return torch.squeeze(loss)

def main():
    device = torch.device("cuda")
    # Step 2: init neural networks
    print("init neural networks")

    vgg16 = models.vgg16(pretrained=True).features
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    vgg16.to(device)
    feature_encoder.to(device)
    relation_network.to(device)

    # Freeze model parameters
    for param in vgg16.parameters():
        param.requires_grad = False

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")
    
    
    transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./cifar-data', train=True,
                                    download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    print("Training...")

    last_accuracy = 0.0
    def train():
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        
        base_dataloader = torch.utils.data.DataLoader(trainset, shuffle=False)

        base, _ = base_dataloader.__iter__().next()

        # calculate features
        sample_features = vgg16_raw(Variable(base).to(device))

        baseline_features = vgg16(Variable(base).to(device)) # n * 512 * 7 * 7

        vgg16_raw.zero_grad()

        for i in range(len(sample_features)): # for each feature vector
            loss = get_loss(sample_features[i], baseline_features[i])
            loss.backward(torch.ones_like(loss))

        feature_encoder_optim.step()

        print("episode:",episode+1)
        print("loss:",torch.sum(loss).item())

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(len(testloader)), 100.*correct/total, correct, total))

    EPISODE = 200
    for episode in range(EPISODE):
        train(episode)
        test()
        scheduler.step()

    print("Done")

if __name__ == '__main__':
    main()
