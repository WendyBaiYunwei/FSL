from dataset import data
import math
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import math
import numpy as np
from torchvision import datasets, models

NUM_EPOCHS = 1

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
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512), ###
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

def adversarial_loss():
    return torch.nn.BCELoss()

def main(): #### to cuda, add in condition
    dataloader = data.get_loader('train') #set batch

    test_samples = []

    teacher = models.vgg16(pretrained=True)
    stu = CNNstudent()
    discriminator = Discriminator()

    
    valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

    gen_opt = torch.optim.Adam(stu.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    print('start training...')
    k = 0
    for epoch in range(NUM_EPOCHS):
        for x, y in dataloader:
            if k < 5:
                test_samples.append(x)
            k += 1
            teacher_embed = teacher(x)
            student_embed = stu(x)

            gen_opt.zero_grad()
            # Generate a batch of images
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(student_embed), valid)
            g_loss.backward()
            gen_opt.step()
            print(g_loss.item())

            dis_opt.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(teacher_embed), valid)
            fake_loss = adversarial_loss(discriminator(student_embed.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            dis_opt.step()
            print(d_loss.item())
            break

    print('start testing...')
    # visualize feature map
    # plot all 64 maps in an 8x8 squares
    feature_maps = [stu(x) for x in test_samples]
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()

if __name__ == '__main__':
    main()