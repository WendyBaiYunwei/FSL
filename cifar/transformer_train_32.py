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

torch.manual_seed(0)
## train: two lenet, mnist, one pretrained, another randomly inited, compare loss
parser = argparse.ArgumentParser()
parser.add_argument("-l","--learning_rate",type = float, default=0.0001) # estimate: 0.2
parser.add_argument("-pre","--pre_trained",type = bool, default=True)
parser.add_argument("-test","--test",type = bool, default=False)
args = parser.parse_args()
LEARNING_RATE = args.learning_rate
pretrained = args.pre_trained
testOnly = args.test
EPOCH = 20
BATCH_SIZE =100
DIM = 32
tokenSize = 4
cropIs = [tokenSize * i for i in range(1, DIM // tokenSize + 1)]

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Linear(DIM ** 2 // 16 * DIM2, 34),                                                   
        )
        self.conv2 = nn.Sequential(         
            nn.Linear(34, 64 * 16),                                        
        )
        self.out = nn.Linear(64 * 16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.out = nn.Linear(64 * 8, 10)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(BATCH_SIZE, -1)
        x = self.out(x)
        return x

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")

transform = transforms.Compose(
            [#transforms.Resize((DIM, DIM)),
                transforms.ToTensor(),
                # transforms.Grayscale(1)
            ])
train_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    transform = transform,
    download = False,
)
test_data = datasets.CIFAR10(
    root = 'data', 
    train = False, 
    transform = transform,
    download = False,
)
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1),
}

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

def getCrops(inputs):
    inputs = inputs.squeeze(1)
    batch = np.zeros((BATCH_SIZE, (DIM ** 2) // (tokenSize ** 2), 3, tokenSize, tokenSize))
    for batchI, input in enumerate(inputs):
        tokenI = 0
        for i in cropIs:
            for j in cropIs:
                token = input[:, i - tokenSize:i, j - tokenSize:j]
                batch[batchI, tokenI, :, :, :] = token
                tokenI += 1
    batch = torch.from_numpy(batch)
    # batch = torch.flatten(batch, start_dim = -2)
    return batch

def train(num_epochs, transformer, loaders, optimizer, classifier):
        transformer.train()
            
        # Train the model
        total_step = len(loaders['train'])
            
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                # images = getCrops(images)
                # images = images.view(BATCH_SIZE, -1)###
                b_x = Variable(images).to(device)   # [batch, 25, 4]
                b_y = Variable(labels).to(device)
                embedding = transformer(b_x.float())   
                output = classifier(embedding)         
                loss = loss_func(output, b_y)
                 
                optimizer.zero_grad()           
                loss.backward()              
                optimizer.step()                
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test(model, classifier):
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            # images = getCrops(images)
            # images = images.view(BATCH_SIZE, -1)###
            embedding = model(Variable(images).to(device).float())
            test_output = classifier(embedding)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.5f' % accuracy)

def main():
    transFn = ''
    classFn = ''    
    classifier = Classifier()
    # classifier.apply(weights_init)
    # classifier.to(device)
    if pretrained == False:
        transformer = TransformerEncoder(dim=tokenSize ** 2,blocks=3,heads=8)
        transFn = 'base_trans.pth'
        classFn = 'base_classifier.pth'
    else:
        model = None
        classFn = 'pre_classifier.pth'
        if testOnly:
            transFn = 'trans_student_separate.pth'
            encoder = Encoder()
            encoder.load_state_dict(torch.load(transFn))
            encoder.to(device)
            classifier.load_state_dict(torch.load(classFn))
            classifier.to(device)
            model = encoder
        else:
            transformer = ResNet50ViT(img_dim=32, pretrained_resnet=True, 
                        blocks=4, classification=False, 
                        dim_linear_block=8, dim=8)
            transFn = 'pre_trans_8.pth'
            classifier.to(device)
            optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
            train(EPOCH, transformer.to(device), loaders, optimizer, classifier)
            torch.save(transformer.state_dict(), transFn)
            torch.save(classifier.state_dict(), classFn)
            model = transformer
        test(model, classifier)

    # optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    # train(EPOCH, transformer.to(device), loaders, optimizer, classifier)
    # torch.save(transformer.state_dict(), transFn)
    # torch.save(classifier.state_dict(), classFn)
    # classifier.load_state_dict(torch.load('./base_classifier.pth'))
    # transformer.load_state_dict(torch.load('./base_trans.pth'))
    # classifier.to(device)
    # transformer.to(device)
    # for param in transformer.parameters():
    #     param.requires_grad = False
    # for param in classifier.parameters():
    #     param.requires_grad = False
    # del transformer
    # transformer.load_state_dict(torch.load('./base_teacher.pth'))
    # for param in transformer.parameters():
    #     param.requires_grad = False

    
    # student = Encoder()
    # student.load_state_dict(torch.load('./trans_student.pth'))
    # optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    # student.to(device)
    # classifier = Classifier()
    # classifier.load_state_dict(torch.load('./base_classifier.pth'))
    # classifier.to(device)
    # train(EPOCH, student, loaders, optimizer, classifier)
    # test(student, classifier)
    print('Done.')

if __name__ == '__main__':
    main()