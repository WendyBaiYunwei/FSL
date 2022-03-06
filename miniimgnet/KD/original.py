# relation net knowledge distillation
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import KD_tg as tg
from torch.autograd import Variable
import numpy as np
import scipy as sp
import scipy.stats
import math
import os
from dataset import get_loader, get_loader_sm
from skimage import io
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR


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
EXPERIMENT_NAME = 'no_KD.txt'
EPISODE = 20000
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 15
teacher_dim = {'channel': 512, 'dim': 7}
stu_dim = {'channel': 64, 'dim': 19}
TEST_EPISODE = 600
T = 20
alpha = 0.1
TEACHER_REL_CLASS = False
TEACHER_NORM_CLASS = False
RESUME_REL_NET = True # 0: scratch, 1:ressume from rel net encoder

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

def getBiggerImg(names):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(224),normalize])
    res = []
    for name in names:
        x_path = os.path.abspath(name)
        x = io.imread(x_path)
        x = transform(x)
        res.append(x)
    return torch.stack(res)

def train(encoder, classifier, classOpt, dim, encOpt):
    last_accuracy = 0.0
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
    losses = []

    enc_sch = StepLR(encOpt,step_size=100000,gamma=0.5)
    cls_sch = StepLR(classOpt,step_size=100000,gamma=0.5)

    encoder.load_state_dict(torch.load(str("./models/stu_enc5way_1shot.pkl")))
    logging.info("load student encoder success")
        
    for episode in range(EPISODE):
        if (episode+1) % 1000 == 0:
            
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)

        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False) ###adjust image dimension, check shuffle
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True) #true
        samples,sample_labels,supportNames = sample_dataloader.__iter__().next()
        batches,batch_labels,batchQueryNames = batch_dataloader.__iter__().next() 

        enc_sch.step(episode)
        cls_sch.step(episode)
        
        # calculate features
        sample_features = encoder(Variable(samples).cuda()) # 5x64*5*5
        # logging.info(sample_features.shape)
        batch_features = encoder(Variable(batches).cuda()) # 20x64*5*5

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #support
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #query
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,dim['channel']*2,dim['dim'],dim['dim']) # teaher / student
        relations = classifier(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda()

        encoder.zero_grad()
        mse = nn.MSELoss().cuda()
        loss = mse(relations,one_hot_labels)

        classifier.zero_grad()

        losses.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm(encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(classifier.parameters(),0.5)

        encOpt.step()
        classOpt.step()

        if (episode+1)%100 == 0:
            logging.info("main episode:" + str(episode+1) + "loss" + str(sum(losses)/len(losses)))
            losses.clear()

        if (episode+1)%5000 == 0:
            logging.info("main Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                num_per_class = 3
                if encOpt: # test student
                    sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)
                    test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True) #true
                else:
                    sample_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=1,split="train",shuffle=False)
                    test_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=num_per_class,split="test",shuffle=True) #true
                    
                sample_images,sample_labels,_ = sample_dataloader.__iter__().next()
                for test_images,test_labels, _ in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = encoder(Variable(sample_images).cuda()) # 5x64
                    test_features = encoder(Variable(test_images).cuda()) # 20x64

                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,dim['channel']*2,dim['dim'],dim['dim'])
                    relations = classifier(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            logging.info("test accuracy:" + str(test_accuracy))

            if test_accuracy > last_accuracy:
                # save networks
                test_res = 'acc: ' + str(test_accuracy) + 'episode: ' + str(episode)
                torch.save(encoder.state_dict(),str("./models/stu_enc"+ EXPERIMENT_NAME + ".pkl"))
                torch.save(classifier.state_dict(),str("./models/stu_class"+ EXPERIMENT_NAME + ".pkl"))
                logging.info('student: '+ test_res)

                logging.info("save networks for episode:" + str(episode))

                last_accuracy = test_accuracy

if __name__ == '__main__': # load existing model
    logging.basicConfig(filename=EXPERIMENT_NAME, level=logging.INFO)

    stuEnc = CNNEncoder()
    stuEnc.apply(weights_init)
    stuEnc.cuda()
    stuClass = StuClassifier()
    stuClass.apply(weights_init)
    stuClass.cuda()
    stuClassifier = RelationNetwork(stu_dim['channel'])
    stuClassifier.apply(weights_init)
    stuClassifier.cuda()

    if RESUME_REL_NET:
        stuEnc.load_state_dict(torch.load(str("./models/stu_enc"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        logging.info("load rel student encoder success")
        stuClassifier.load_state_dict(torch.load(str("./models/stu_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        logging.info("load rel student classifier success")
    
    enc_optimizer = torch.optim.Adam(stuEnc.parameters(),lr=LEARNING_RATE)
    
    cls_optimizer = torch.optim.Adam(stuClassifier.parameters(),lr=LEARNING_RATE)
    train(stuEnc, stuClassifier, cls_optimizer, stu_dim, enc_optimizer)