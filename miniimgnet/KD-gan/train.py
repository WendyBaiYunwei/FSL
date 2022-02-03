from dataset import get_loader, get_loader_sm
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models
from skimage import io, img_as_ubyte

NUM_EPOCHS = 5
img_shape = (512, 7, 7)
Tensor = torch.cuda.FloatTensor 
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
b1 = 0.5
b2 = 0.999
torch.manual_seed(0)
adversarial_loss = torch.nn.BCELoss()
CLASS_NUM = 64

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
        self.inter1 = nn.Linear(21 * 21 * 64, 5000)
        self.dropout = nn.Dropout(p=0.8)
        self.inter2 = nn.Linear(5000, 7 * 7 * 512 + CLASS_NUM)
        # self.out = nn.Linear(7 * 7 * 512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim = 1)
        x = self.inter1(x)
        x = self.inter2(self.dropout(x))
        # y = self.out(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
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

def main(): #### add in condition
    dataloader = get_loader() #set batch
    dataloader_sm = get_loader_sm()

    teacher = models.vgg16(pretrained=True).features
    stu = CNNstudent()
    discriminator = Discriminator()

    teacher.cuda()
    stu.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
    gen_opt = torch.optim.Adam(stu.parameters(), lr=LEARNING_RATE, betas=(b1, b2))
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(b1, b2))

    print('start training...')
    k = 0
    
    dataloader_sm = iter(dataloader_sm)
    for epoch in range(NUM_EPOCHS):
        loss = [0, 0]
        for x, y in dataloader:
            x_sm, _ = next(dataloader_sm)
            valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

            if k == 0:
                test_samples = x
                test_samples_sm = x_sm

            if (k + 1) % 100 == 0:
                print(k)
                print('epoch', epoch, 'gen loss', loss[0] / k)
                print('dis loss', loss[1] / k)

            k += 1
            teacher_embed = teacher(x.cuda())
            student_embed = stu(x_sm.cuda())

            gen_opt.zero_grad()
            # Generate a batch of images
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(student_embed), valid)
            g_loss.backward()
            gen_opt.step()
            loss[0] += g_loss.item()
            

            dis_opt.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(torch.flatten(teacher_embed, start_dim = 1)), valid)
            fake_loss = adversarial_loss(discriminator(student_embed.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            dis_opt.step()
            loss[1] += d_loss.item()

        print('epoch', epoch, 'gen loss', loss[0] / len(dataloader))
        print('dis loss', loss[1] / len(dataloader))

    print('saving model...')
    # torch.save(stu.state_dict(),str("./models/kdgan_stu.pkl"))
    # torch.save(discriminator.state_dict(),str("./models/kdgan_discriminator.pkl"))

    print('start testing...')
    # visualize feature map
    # plot all 64 maps in an 8x8 squares
    
    teacher_embedding = teacher(test_samples.cuda())
    # print(teacher_embedding.shape)
    stu_embedding = stu(test_samples_sm.cuda())
    stu_embedding = stu_embedding.view(-1, 512, 7, 7)
    arr = [teacher_embedding.detach().cpu(), stu_embedding.detach().cpu()]
    for i in range(5):
        for j in range(2):
            # specify subplot and turn of axis
            # ax = plt.subplot(1, 2, j + 1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # plot filter channel in grayscale
            if j == 0:
                io.imsave('teacher' + str(i) + '.png', arr[j][0][i, :, :])
            else:
                io.imsave('stu' + str(i) + '.png', arr[j][0][i, :, :])

if __name__ == '__main__':
    main()