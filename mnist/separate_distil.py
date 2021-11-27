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
import numpy as np
import argparse
import math

## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()
torch.manual_seed(0)

LEARNING_RATE = args.learning_rate ####
EPOCH = 5
BATCH_SIZE = 10
DIM = 8
DIM2 = 4

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,           
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 8, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(8 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = x.view(100, -1)
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden1 = nn.Linear(DIM * DIM // 16 * DIM2, 4)
        self.classifier = nn.Linear(4, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden1(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_loss(out, target):
    # 49, 512
    loss = torch.square(out - target)
    # loss = torch.squeeze(loss)
    return loss
    
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
    student = CNN()
    student.apply(weights_init)
    student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    # scheduler = StepLR(optimizer,step_size=1,gamma=0.98)

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ])

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transform,
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)
    # trainloader_sm = torch.utils.data.DataLoader(train_data_sm, 
    #                                     batch_size=BATCH_SIZE, 
    #                                     shuffle=False, 
    #                                     num_workers=1)

    print("Training...")
    
    student.train()

    def train():
        epoch_loss = 0
        count = 0
        # dataiter_sm = iter(trainloader_sm)
        for inputs, _ in trainloader:
            sample = student(Variable(inputs).to(device))
            
            baseline_features = teacher_trans(Variable(inputs).to(device).float()) # 16 * 32 * 7 * 7
            baseline = teacher_cls(Variable(baseline_features).to(device))[0].item()
            # inputs_sm, _ = next(dataiter_sm)
            optimizer.zero_grad()

            loss = get_loss(sample, baseline)

            loss.backward()

            optimizer.step()

            epoch_loss += torch.sum(loss).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1

    for episode in range(EPOCH):
        train()
        # scheduler.step()
        torch.save(student.state_dict(), './trans_student_separate.pth')
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()