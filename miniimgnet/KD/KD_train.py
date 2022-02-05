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


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
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

class CNNstudent(nn.Module):
    def __init__(self):
        super(CNNstudent, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,           
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),      
            nn.Conv2d(32, 64, 3, 1, 1),     
            nn.ReLU(),
        )
        self.inter1 = nn.Linear(19 * 19 * 64, 5000)
        self.inter2 = nn.Linear(5000, 7 * 7 * 512)
        self.out = nn.Linear(7 * 7 * 512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim = 1)
        x = self.inter1(x)
        x = self.inter2(x)
        y = self.out(x)
        return x, y
        
torch.manual_seed(0)
LEARNING_RATE = 0.0005
EXPERIMENT_NAME = 'TEST.txt'
EPISODE = 1
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 15
teacher_dim = {'channel': 512, 'dim': 7}
stu_dim = {'channel': 64, 'dim': 19}
FEATURE_DIM = 64
TEST_EPISODE = 600
T = 20
alpha = 0.9

torch.manual_seed(0)

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
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def loss_fn_kd(outputs, labels, teacher_outputs):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def train(encoder, classifier, classOpt, dim, encOpt = None, teacher_encoder = None, teacher_classifier = None):
    for episode in range(EPISODE):
        metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)

        if encOpt: # if training the student
            sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False) ###adjust image dimension, check shuffle
            batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True) #true
            sample_dataloader_big = tg.get_mini_imagenet_data_loader_big(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
            batch_dataloader_big = tg.get_mini_imagenet_data_loader_big(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True) #true
            samples2,sample_labels2,supportNames2 = sample_dataloader_big.__iter__().next()
            batches2,batch_labels2,batchQueryNames2 = batch_dataloader_big.__iter__().next()
        else: # trains teacher
            sample_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
            batch_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True) #true
        
        # sample datas
        samples,sample_labels,supportNames = sample_dataloader.__iter__().next()
        batches,batch_labels,batchQueryNames = batch_dataloader.__iter__().next()
        print('teacher: ', sample_labels2, supportNames2, batch_labels2, batchQueryNames2)
        print('student: ', sample_labels, supportNames, batch_labels, batchQueryNames)
        # exit()
        
        
        # calculate features
        sample_features = encoder(Variable(samples).cuda()) # 5x64*5*5
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

            loss = loss_fn_kd(relations, one_hot_labels, teacher_outputs)
        else:
            mse = nn.MSELoss().cuda()
            loss = mse(relations,one_hot_labels)

        classifier.zero_grad()

        loss.backward()

        if encOpt:
            torch.nn.utils.clip_grad_norm(encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(classifier.parameters(),0.5)

        if encOpt:
            encOpt.zero_grad()
        classOpt.step()

        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",loss.item())

        if (episode+1)%5000 == 0:
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
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    relations = classifier(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy
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

if __name__ == '__main__': # load existing model
    logging.basicConfig(filename=EXPERIMENT_NAME, level=logging.INFO)
    teacherEnc = models.vgg16(pretrained=True).features
    for param in teacherEnc.params():
        param.require_gradient = False
    teacherClassifier = RelationNetwork()
    teacherClassifier.apply(weights_init)
    tempOpt = torch.optim.Adam(teacherClassifier.parameters(),lr=LEARNING_RATE)
    train(teacherEnc, teacherClassifier, tempOpt, teacher_dim)
    for param in teacherClassifier.params():
        param.require_gradient = False

    stuEnc = CNNstudent()
    stuClassifier = RelationNetwork()
    stuEnc.apply(weights_init)
    stuClassifier.apply(weights_init)
    enc_optimizer = torch.optim.Adam(stuEnc.parameters(),lr=LEARNING_RATE)
    cls_optimizer = torch.optim.Adam(stuClassifier.parameters(),lr=LEARNING_RATE)
    train(stuEnc, stuClassifier, cls_optimizer, stu_dim, enc_optimizer, teacherEnc, teacherClassifier)