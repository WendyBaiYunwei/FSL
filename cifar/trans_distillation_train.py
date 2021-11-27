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
parser.add_argument("-l","--learning_rate",type = float, default=0.0001) # estimate: 0.2
args = parser.parse_args()
torch.manual_seed(0)

LEARNING_RATE = args.learning_rate
EPOCH = 5
BATCH_SIZE = 1
DIM = 32
tokenSize = 4
cropIs = [tokenSize * i for i in range(1, DIM // tokenSize + 1)]
transFn = 'pre_trans_8.pth'
classFn = 'pre_classifier.pth'

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Linear(32 * 32 * 3, 10),                                                   
        )
        self.conv2 = nn.Sequential(         
            nn.Linear(10, 64 * 8),                                        
        )
        self.out = nn.Linear(64 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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
    
    teacher = ResNet50ViT(img_dim=32, pretrained_resnet=True, 
                        blocks=4, classification=False, 
                        dim_linear_block=8, dim=8)
    teacher.load_state_dict(torch.load(transFn))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)
    student = Encoder()
    student.apply(weights_init).to(device)
    student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer,step_size=1,gamma=0.98)

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ])

    # train_data_sm = datasets.MNIST(
    #     root = 'data',
    #     train = True,                         
    #     transform = transform,
    #     download = False,            
    # )

    train_data = datasets.CIFAR10(
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

    def train(episode):
        epoch_loss = 0
        count = 0
        # dataiter_sm = iter(trainloader_sm)
        for inputs, _ in trainloader:
            inputs_sm = torch.flatten(inputs, start_dim = 1)
            sample_features = student(Variable(inputs_sm).to(device))

            baseline_features = teacher(Variable(inputs).to(device).float()).flatten(start_dim = 1) # 16 * 32 * 7 * 7
            # inputs_sm, _ = next(dataiter_sm)
            

            optimizer.zero_grad()

            loss = get_loss(sample_features, baseline_features)

            loss.backward(torch.ones_like(sample_features))

            optimizer.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1

    for episode in range(EPOCH):
        train(episode)
        scheduler.step()
        torch.save(student.state_dict(), './trans_student_4lr.pth')
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()