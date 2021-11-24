import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
from self_attention_cv import TransformerEncoder
import numpy as np
import cv2
import argparse
import math
import argparse


## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.01) # estimate: 0.2
args = parser.parse_args()

LEARNING_RATE = args.learning_rate
EPOCH = 10
BATCH_SIZE = 100
DIM = 28
cropSize = 4
cropIs = [DIM // cropSize * i for i in range(1, cropSize + 1)]
tokenSize = (DIM // cropSize) ** 2
torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Linear(28 * 28, 14),                                                   
        )
        self.conv2 = nn.Sequential(         
            nn.Linear(14, 49 * 16),                                        
        )
        self.out = nn.Linear(49 * 16, 10)

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

def getCrops(inputs):
    inputs = inputs.squeeze()
    # batch, 28, 28
    batch = np.zeros((BATCH_SIZE, tokenSize, cropSize, cropSize))
    for batchI, input in enumerate(inputs):
        tokenI = 0
        for i in cropIs:
            for j in cropIs:
                token = input[i - cropSize:i, j - cropSize:j]
                batch[batchI, tokenI, :, :] = token
                tokenI += 1
    batch = torch.from_numpy(batch)
    batch = torch.flatten(batch, start_dim = -2)
    return batch
    
def main():
    device = torch.device("cuda")
    
    teacher = TransformerEncoder(dim=16,blocks=3,heads=4)####
    teacher.load_state_dict(torch.load('./base_trans.pth'))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)
    student = Encoder()
    student.apply(weights_init).to(device)
    student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer,step_size=1,gamma=0.9)

    # transform = transforms.Compose(
    #             [transforms.Resize((13, 13)),
    #                 transforms.ToTensor(),
    #             ])

    # train_data_sm = datasets.MNIST(
    #     root = 'data',
    #     train = True,                         
    #     transform = transform,
    #     download = False,            
    # )

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transforms.ToTensor(),
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

            inputs = getCrops(inputs)
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
        torch.save(student.state_dict(), './trans_student.pth')
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()