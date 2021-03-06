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
import numpy as np
import KD_tg as tg
import math
import argparse
import scipy as sp
import scipy.stats
from networks.res12 import ResNet
from torchvision import models
import neptune.new as neptune

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 640)#td, 512
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 5)
parser.add_argument("-e","--episode",type = int, default= 100000) #500000 ####
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=str, default='cuda:0')
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-n","--name",type=str,default='51-res12-baseline')
parser.add_argument("-nt","--type",type=str,default='resnet12')
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
EXPERIMENT_NAME = args.name
NET_TYPE = args.type
DIM = 3
DIM2 = 14
if NET_TYPE == 'resnet18':
    DIM = 3
    DIM2 = 7
    FEATURE_DIM = 512

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,in_channel, hidden_size=8):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channel*2,256,kernel_size=1,padding=0),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(256,128,kernel_size=1,padding=0),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)) #td
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=1,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64*DIM*DIM,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
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

def main():
    run = neptune.init(
        project="ywb/kd-brenaic",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4Yjk0ZDlkMi04ZTBhLTQ4YzktYWE2Ni02Njg0OGQwOWFiNjkifQ==",
    ) 
    params = {"name": EXPERIMENT_NAME}
    print(EXPERIMENT_NAME)
    run["parameters"] = params
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")

    if args.type == 'resnet12':
        feature_encoder = ResNet()
        model_dict = feature_encoder.state_dict()        
        pretrained_dict = torch.load("../models/Res12-pre.pth")['params']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        feature_encoder.load_state_dict(model_dict)    
    else:
        resnet18 = models.resnet18(pretrained=True)
        modules=list(resnet18.children())[:-2]
        feature_encoder=nn.Sequential(*modules)
        for param in feature_encoder.parameters():
            param.requires_grad = False
    relation_network = RelationNetwork(FEATURE_DIM)
    feature_encoder.to(device=GPU)
    relation_network.to(device=GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    losses = []

    for episode in range(EPISODE):
        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS) #td
        sample_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=SAMPLE_NUM_PER_CLASS,\
            split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=BATCH_NUM_PER_CLASS,\
            split="test",shuffle=True) #true

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()
        
        # calculate features
        sample_features = feature_encoder(Variable(samples).to(device=GPU)) # 5x640*5*5
        batch_features = feature_encoder(Variable(batches).to(device=GPU)) # 20x640*5*5
        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #support
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) #query
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,DIM2,DIM2) #td, 55
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        mse = nn.MSELoss().to(device=GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1,\
            batch_labels.view(-1,1), 1)).to(device=GPU)
        loss = mse(relations,one_hot_labels)

        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%100 == 0:
            inf = str(sum(losses)/len(losses))
            print(episode, inf)
            run["loss"].log(inf)
            losses.clear()

        if (episode+1)%5000 == 0: #5000
            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=1,split="train",shuffle=False)

                num_per_class = 3
                test_dataloader = tg.get_mini_imagenet_data_loader_big(task,num_per_class=num_per_class,split="test",\
                    shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).to(device=GPU)) # 5x64
                    test_features = feature_encoder(Variable(test_images).to(device=GPU)) # 20x64

                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,DIM2,DIM2)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

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
                torch.save(feature_encoder.state_dict(),str("./models/miniimagenet_feature_encoder_" +\
                     EXPERIMENT_NAME +".pkl"))
                torch.save(relation_network.state_dict(),str("./models/miniimagenet_relation_network_" +\
                    EXPERIMENT_NAME +".pkl"))

                print("save networks for episode:",episode)

                inf = str(episode) + ' ' + str(test_accuracy)
                print(inf)
                run["val_acc"].log(inf)
                last_accuracy = test_accuracy
    run.stop()

if __name__ == '__main__':
    main()
