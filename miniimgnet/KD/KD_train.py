# relation net knowledge distillation
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import task_generator as tg
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
EXPERIMENT_NAME = 'alpha0.9.txt'
EPISODE = 500000
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 15
teacher_dim = {'channel': 512, 'dim': 7}
stu_dim = {'channel': 64, 'dim': 19}
TEST_EPISODE = 600
T = 20
alpha = 0.9 #to-do change / 29.7
SKIP_TEACHER = True
STU_ENC_TYPE = 2 # 0: scratch, 1: norm encoder, 2:ressume from rel net encoder

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

def train(encoder, classifier, classOpt, dim, encOpt = None, teacher_encoder = None, teacher_classifier = None):
    last_accuracy = 0.0
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
    losses = []
    if encOpt:
        enc_sch = StepLR(encOpt,step_size=100000,gamma=0.5)
        cls_sch = StepLR(classOpt,step_size=100000,gamma=0.5)
    for episode in range(EPISODE):
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)

        if encOpt: # if training the student
            enc_sch.step(episode)
            cls_sch.step(episode)
            sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False) ###adjust image dimension, check shuffle
            batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True) #true
            samples,sample_labels,supportNames = sample_dataloader.__iter__().next()
            batches,batch_labels,batchQueryNames = batch_dataloader.__iter__().next()
            samples2 = getBiggerImg(supportNames)
            batches2 = getBiggerImg(batchQueryNames)
        else: # trains teacher
        # sample datas
            _,sample_labels,supportNames = sample_dataloader.__iter__().next()
            _,batch_labels,batchQueryNames = batch_dataloader.__iter__().next()
            samples = getBiggerImg(supportNames)
            batches = getBiggerImg(batchQueryNames) #true
        
        
        # calculate features
        sample_features = encoder(Variable(samples).cuda()) # 5x64*5*5
        # print(sample_features.shape)
        batch_features = encoder(Variable(batches).cuda()) # 20x64*5*5

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #support
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #query
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,dim['channel']*2,dim['dim'],dim['dim']) # teaher / student
        relations = classifier(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda()

        if encOpt:
            encoder.zero_grad()
            sample_features = teacher_encoder(Variable(samples2).cuda()) # 5x64*5*5
            batch_features = teacher_encoder(Variable(batches2).cuda())
            teacher_support_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #support
            teacher_q_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #query
            teacher_q_features_ext = torch.transpose(teacher_q_features_ext,0,1)
            teaccher_relation_pairs = torch.cat((teacher_support_features_ext,teacher_q_features_ext),2).view(-1,teacher_dim['channel']*2,teacher_dim['dim'],teacher_dim['dim'])
            teacher_outputs = teacher_classifier(teaccher_relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

            loss = loss_fn_kd(relations, batch_labels, teacher_outputs)
        else:
            mse = nn.MSELoss().cuda()
            loss = mse(relations,one_hot_labels)

        classifier.zero_grad()

        losses.append(loss.item())
        loss.backward()

        if encOpt:
            torch.nn.utils.clip_grad_norm(encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(classifier.parameters(),0.5)

        if encOpt:
            encOpt.step()
        classOpt.step()

        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",sum(losses)/len(losses))
            losses.clear()

        if (episode+1)%500 == 0:
            print("Testing...")
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

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:
                # save networks
                test_res = 'acc: ' + str(test_accuracy) + 'episode: ' + str(episode)
                if encOpt:
                    torch.save(encoder.state_dict(),str("./models/stu_enc"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save(classifier.state_dict(),str("./models/stu_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    logging.info('student: '+ test_res)
                else:
                    torch.save(classifier.state_dict(),str("./models/teacher_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    logging.info('teacher: '+ test_res)

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy

def test(enc, classifier, type):
    print('testing...')
    if type == 'teacher':
        testLoader = get_loader('test')
    else:
        testLoader = get_loader_sm('test')
    enc.eval()
    classifier.eval()
    accuracy = 0
    count = 0
    for inputs, labels, _ in testLoader:
        x = enc(Variable(inputs).cuda())
        output = classifier(x)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        labels = Variable(labels).cuda()
        accuracy += (pred_y == labels).sum().item()
        count += 1
    print('Test Accuracy of the model on the test images:', accuracy  / 600 / 20)
    return accuracy

def traditionalKD(stuEnc, stuClass, teacherEnc):
    trainLoader = get_loader('train')
    teacherClass = TeacherClassifier()
    teacherClass.apply(weights_init)
    teacherClass.cuda()
    optimizer = torch.optim.Adam(teacherClass.parameters(), lr=1e-3)
    lFunc = nn.CrossEntropyLoss()

    best_acc = 0
    # train teacher classifier
    print('train teacher classifier...')
    for epoch in range(30):
        print(epoch)
        for x, y, _ in trainLoader:
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
    trainLoader = get_loader_sm('train')
    trainLoaderBig = get_loader
    stuEncOpt = torch.optim.Adam(stuEnc.parameters(), lr=1e-3)
    stuClassOpt = torch.optim.Adam(stuClass.parameters(), lr=1e-3)
    print('train student encoder...')
    for epoch in range(30):
        print(epoch)
        for x, y, paths in trainLoader:
            input = stuEnc(Variable(x).cuda())
            outputs = stuClass(input)

            x = getBiggerImg(paths)
            x = teacherEnc(Variable(x).cuda())
            teacher_outputs = teacherClass(x)
            stuEncOpt.zero_grad()
            stuClassOpt.zero_grad()

            label = Variable(y).cuda()
            loss = loss_fn_kd(outputs, label, teacher_outputs)
            loss.backward()

            stuEncOpt.step()
            stuClassOpt.step()

        acc = test(stuEnc, stuClass, 'student')
        if acc > best_acc:
            # save teacher classifier
            torch.save(stuEnc.state_dict(),str("./models/stu_enc_norm"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
            best_acc = acc
    
    stuEnc.load_state_dict(torch.load(str("./models/stu_enc_norm"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    return stuEnc

if __name__ == '__main__': # load existing model
    logging.basicConfig(filename=EXPERIMENT_NAME, level=logging.INFO)

    resnet18 = models.resnet18(pretrained=True)
    modules=list(resnet18.children())[:-2]
    teacherEnc=nn.Sequential(*modules)
    teacherEnc.cuda()
    for param in teacherEnc.parameters():
        param.requires_grad = False

    stuEnc = CNNEncoder()
    stuEnc.apply(weights_init)
    stuEnc.cuda()
    stuClass = StuClassifier()
    stuClass.apply(weights_init)
    stuClass.cuda()
    stuClassifier = RelationNetwork(stu_dim['channel']) #to-do: scheduler
    stuClassifier.apply(weights_init)
    stuClassifier.cuda()

    if STU_ENC_TYPE == 0:
        stuEnc = traditionalKD(stuEnc, stuClass, teacherEnc)
    elif STU_ENC_TYPE == 1 and os.path.exists(str("./models/stu_enc_norm" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        stuEnc.load_state_dict(torch.load(str("./models/stu_enc_norm"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load student encoder success")
    else:
        stuEnc.load_state_dict(torch.load(str("./models/stu_enc"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load student encoder success")
        stuClassifier.load_state_dict(torch.load(str("./models/stu_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load student relation classifier success")
    
    teacherClassifier = TeacherRelationNetwork(teacher_dim['channel'])
    teacherClassifier.apply(weights_init)
    teacherClassifier.cuda()
    
    tempOpt = torch.optim.Adam(teacherClassifier.parameters(),lr=LEARNING_RATE)
    if SKIP_TEACHER and os.path.exists(str("./models/teacher_class" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        teacherClassifier.load_state_dict(torch.load(str("./models/teacher_class"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load teacher relation network success")
    else:
        print('Prepare teacher relation network...')
        train(teacherEnc, teacherClassifier, tempOpt, teacher_dim)

    for param in teacherClassifier.parameters():
        param.requires_grad = False

    enc_optimizer = torch.optim.Adam(stuEnc.parameters(),lr=LEARNING_RATE)
    
    cls_optimizer = torch.optim.Adam(stuClassifier.parameters(),lr=LEARNING_RATE)
    
    train(stuEnc, stuClassifier, cls_optimizer, stu_dim, enc_optimizer, teacherEnc, teacherClassifier)