# relation net knowledge distillation
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import KD_tg as tg
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import scipy as sp
import scipy.stats
import math
import os
from skimage import io
import cv2
from dataset import get_loader, get_loader_sm

class TeacherClassifier(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,hidden_size=20):
        super(TeacherClassifier, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,256,kernel_size=1,padding=0),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(256,128,kernel_size=1,padding=0),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,32,kernel_size=1,padding=0),
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(32*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,64)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class StuClassifier(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,hidden_size=20):
        super(StuClassifier, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64,32,kernel_size=3,padding=0),
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,32,kernel_size=3,padding=0),
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(32*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,64)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size=8):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=0),
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

class TeacherRelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,hidden_size=8):
        super(TeacherRelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=1,padding=0),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(512,256,kernel_size=1,padding=0),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv2d(256,64,kernel_size=1,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
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
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64
        
torch.manual_seed(0)
LEARNING_RATE = 0.001
EXPERIMENT_NAME = '3-3-simpleKD'
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 15
teacher_dim = {'channel': 512, 'dim': 7}
stu_dim = {'channel': 64, 'dim': 19}
TEST_EPISODE = 600
BATCH_SIZE = 32
T = 20
alpha = 0.9
TEACHER_REL_CLASS = False
TEACHER_NORM_CLASS = True
RESUME_REL_NET = False # 0: scratch, 1:ressume from rel net encoder
EPOCHS = 30

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def loss_fn_kd(outputs, labels, teacher_outputs):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels.long().cuda()) * (1. - alpha)

    return KD_loss

def getBiggerImg(imgs):
    res = []
    for img in imgs:
        img = np.swapaxes(img, 0, 1) #3,84,84 -> 84,84,3 -> 3,84,84
        img = np.swapaxes(img, 1, 2) 
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img = np.swapaxes(img, 1, 2) #3,84,84 -> 84,84,3 -> 3,84,84
        img = np.swapaxes(img, 0, 1) 
        res.append(img)
    return torch.from_numpy(np.array(res))

def test(enc, classifier, type):
    inf = 'testing on normal classifier...'
    print(inf)
    logging.info(inf)

    if type == 'teacher':
        testLoader = get_loader('test')
    else:
        testLoader = get_loader_sm('test')
    enc.eval()
    classifier.eval()
    accuracy = 0
    count = 0
    for inputs, labels in testLoader:
        x = enc(Variable(inputs).cuda())
        output = classifier(x)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        labels = Variable(labels).cuda()
        accuracy += (pred_y == labels).sum().item()
        count += 1
    inf = 'Test Accuracy of the model on the test images (normal):' + str(accuracy / 600 / 20)
    print(inf)
    logging.info(inf)
    return accuracy

def traditionalKD(stuEnc, stuClass, teacherEnc, teacherClass):
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
    trainloader_lg = get_loader('train')
    
    lFunc = nn.CrossEntropyLoss()

    best_acc = 0
    # train teacher classifier
    if TEACHER_NORM_CLASS:
        inf = 'train teacher normal classifier...'
        print(inf)
        logging.info(inf)
        for epoch in range(EPOCHS):
            logging.info(str(epoch))
            for x, y in trainloader_lg:
                x = teacherEnc(Variable(x).cuda())
                output = teacherClass(x)
                optimizer.zero_grad()

                label = Variable(y).cuda()
                loss = lFunc(output, label)
                loss.backward()

                optimizer.step()
            
            acc = test(teacherEnc, teacherClass, 'teacher')
            if acc > best_acc:
                # save teacher classifier
                torch.save(teacherClass.state_dict(),str("./models/teacher_norm_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                best_acc = acc

    for param in teacherClass.parameters():
        param.requires_grad = False
    # load teacher class
    teacherClass.load_state_dict(torch.load(str("./models/teacher_norm_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    
    # train student using distillation
    # test student
    best_acc = 0
    stuEncOpt = torch.optim.Adam(stuEnc.parameters(), lr=1e-3)
    stuClassOpt = torch.optim.Adam(stuClass.parameters(), lr=1e-3)

    inf = 'train student encoder via normal...'
    print(inf)
    logging.info(inf)
    losses = []
    trainloader = get_loader_sm('train')
    for epoch in range(EPOCHS):
        logging.info(str(epoch))
        for x, y in trainloader:
            input = stuEnc(Variable(x).cuda())
            outputs = stuClass(input)

            x = getBiggerImg(x.numpy())
            x = teacherEnc(Variable(x).cuda())
            teacher_outputs = teacherClass(x)
            stuEncOpt.zero_grad()
            stuClassOpt.zero_grad()

            label = Variable(y).cuda()
            loss = loss_fn_kd(outputs, label, teacher_outputs)
            losses.append(loss.item())
            loss.backward()

            stuEncOpt.step()
            stuClassOpt.step()
        inf = 'train_loss: ' + str(sum(losses) / len(losses))
        print(inf)
        logging.info(inf)  
        if epoch > 20:
            acc = test(stuEnc, stuClass, 'student')
            if acc > best_acc:
                torch.save(stuEnc.state_dict(),str("./models/" + EXPERIMENT_NAME + "-stuEnc.pkl"))
                best_acc = acc

if __name__ == '__main__': # load existing model
    logging.basicConfig(filename=EXPERIMENT_NAME + '.txt', level=logging.INFO)

    resnet18 = models.resnet18(pretrained=False)
    modules=list(resnet18.children())[:-2]
    teacherEnc=nn.Sequential(*modules)
    teacherEnc.load_state_dict(torch.load('./models/teacher_enc.pkl'))
    teacherEnc.cuda()
    for param in teacherEnc.parameters():
        param.requires_grad = False

    stuEnc = CNNEncoder()
    stuEnc.apply(weights_init)
    stuEnc.cuda()
    stuClass = StuClassifier()
    stuClass.apply(weights_init)
    stuClass.cuda()
    
    teacherClass = TeacherClassifier()
    teacherClass.apply(weights_init)
    teacherClass.cuda()
    optimizer = torch.optim.Adam(teacherClass.parameters(), lr=1e-3)
    traditionalKD(stuEnc, stuClass, teacherEnc, teacherClass)
    
