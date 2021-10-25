# if classifier doesn't yet exist:
#   stage 1: transfer learning on vgg16 for mnist -> obtain classifier for mnist
# stage 2: test encoder (version N) with the classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim import Adam
from torchvision import datasets, models
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from my_fc import MyLinearLayer
import logging
from datetime import datetime
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version_number", type = str, required=True)
args = parser.parse_args()
VERSION = args.version_number
TEST_CLASSIFIER = True
EPOCH = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.0005

def prepare_dataset():
    transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                    transforms.ToTensor(), ##normalize
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
    trainset = datasets.MNIST(root='./datas/mnist', download=True, transform=transform)
    testset = datasets.MNIST(root='./datas/mnist', train = False, download=True, transform=transform)
    return trainset, testset

def transfer_learning(trainset):
    # prepare model
    vgg16 = None
    if os.path.exists('./vgg16.pth'):
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('./vgg16.pth'))
    else:
        vgg16 = models.vgg16(pretrained=True)
    
    for param in vgg16.parameters():
        param.requires_grad = False
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, 10)
    for param in vgg16.classifier[6].parameters():
        param.requires_grad = True
    vgg16 = vgg16.cuda()

    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=True, batch_size = BATCH_SIZE)
    
    print("Training...")
    criterion = CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = Adam(vgg16.classifier[6].parameters(), lr=LEARNING_RATE)

    def train(epoch):
        training_loss = []
        i = 0
        for inputs, labels in trainloader:
            inputs = inputs.repeat(1, 3, 1, 1) ## change to transforms

            optimizer.zero_grad()
            out = vgg16(Variable(inputs).cuda())
            labels = Variable(labels).cuda()

            loss = criterion(out, labels)
            training_loss.append(loss.item())

            if i % 10000 == 0:
                print(epoch + 1, np.average(training_loss))

            loss.backward()
            optimizer.step()

            i += 1
        return np.average(training_loss)

    for epoch in range(EPOCH):
        epoch_loss = train(epoch)
        torch.save(vgg16.state_dict(), './transfer_mnist_vgg16.pth')
        now = datetime.now()
        current_time = now.strftime("%m/%d-%H:%M:%S")
        logging.info("train episode:" + str(epoch+1) + " training loss:" +\
             str(epoch_loss / len(trainloader)) +\
            str(current_time))
    return vgg16

def test(testset, classifier, feature):
    # test
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=True, batch_size = BATCH_SIZE)

    prediction = []
    target = []
    feature = feature.cuda()
    classifier = classifier.cuda()
    for inputs, labels in testloader:
        inputs = inputs.repeat(1, 3, 1, 1)
        with torch.no_grad():
            output1 = feature(inputs.cuda()) #output: 512,7,7
            output1 = torch.flatten(output1, start_dim = 1)
            output = classifier(output1.cuda())

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, 1)
        prediction.append(predictions)
        target.append(labels)
    
    accuracy = []
    for i in range(len(prediction)):
        accuracy.append(accuracy_score(target[i], prediction[i]))

    print('testing accuracy:', np.average(accuracy))
    now = datetime.now()
    current_time = now.strftime("%m/%d-%H:%M:%S")
    logging.info('testing accuracy: '+ str(np.average(accuracy)) +\
        ' ' + str(current_time))

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

if __name__ =='__main__':
    logging.basicConfig(filename='mnist_test' + VERSION +'.log', level=logging.INFO)

    now = datetime.now()

    current_time = now.strftime("%m/%d-%H:%M:%S")
    logging.info(" Current Time = " + current_time)
    logging.info(VERSION + '-')

    trainset, testset = prepare_dataset()

    if os.path.exists('./transfer_mnist_vgg16.pth'):
        vgg16 = models.vgg16(pretrained=False)
        num_ftrs = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_ftrs, 10)
        vgg16.load_state_dict(torch.load('./transfer_mnist_vgg16.pth'))
    else:
        vgg16 = transfer_learning(trainset)
    if TEST_CLASSIFIER:
        test(testset, vgg16.classifier, vgg16.features)
    else:
        encoder = CNNEncoder()
        encoder.load_state_dict(torch.load('./feature_encoder_mnist' + VERSION + '.pth'))
        test(testset, vgg16.classifier, encoder)
    
    print('Done')