import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
from self_attention_cv import TransformerEncoder, ResNet50ViT
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import argparse
import math

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss

torch.manual_seed(0)

BATCH_SIZE = 100
DIM = 28
DIM2 = 6

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden1 = nn.Linear(DIM * DIM // 16 * DIM2, 4)
        self.classifier = nn.Linear(4, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden1(x)
        return x
    
def main():
    device = torch.device("cuda")
    teacher_trans = ResNet50ViT(img_dim=DIM, pretrained_resnet=True, 
                        blocks=3, classification=False, 
                        dim_linear_block=DIM2, dim=DIM2)
    teacher_trans.load_state_dict(torch.load('./base_trans.pth'))
    teacher_trans.to(device)
    for param in teacher_trans.parameters():
        param.requires_grad = False
    teacher_cls = Classifier()
    teacher_cls.load_state_dict(torch.load('./base_classifier.pth'))
    for param in teacher_cls.parameters():
        param.requires_grad = False
    teacher_cls = teacher_cls.hidden1
    teacher_cls.to(device)

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ])

    train_data = datasets.MNIST(
        root = 'data',
        train = False,                         
        transform = transforms.ToTensor(),
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)

    print("Getting labels...")

    def getLabels():
        count = 0
        # dataiter_sm = iter(trainloader_sm)
        labels = []
        for inputs, _ in trainloader:
            teacherIn = inputs.repeat(1, 3, 1, 1)
            baseline_features = teacher_trans(Variable(teacherIn).to(device).float()) # 16 * 32 * 7 * 7
            baseline_features = baseline_features.flatten(start_dim = 1)
            baseline = teacher_cls(Variable(baseline_features).to(device))
            labels.append(np.array(baseline.cpu()))
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        return labels

    labels = getLabels()
    torch.save(labels, 'labels_test.pt')
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()