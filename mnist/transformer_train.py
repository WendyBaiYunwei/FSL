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

torch.manual_seed(0)
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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.out = nn.Linear(49 * 16, 10)

    def forward(self, x):
        x = x.view(BATCH_SIZE, -1)
        x = self.out(x)
        return x

loss_func = nn.CrossEntropyLoss() 
device = torch.device("cuda")

transform = transforms.Compose(
            [#transforms.Resize((13, 13)),
                transforms.ToTensor(),
            ])
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = transform
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transform
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

def train(num_epochs, transformer, loaders, optimizer, classifier):
        transformer.train()
            
        # Train the model
        total_step = len(loaders['train'])
            
        for epoch in range(num_epochs):####
            for i, (images, labels) in enumerate(loaders['train']):
                images = getCrops(images)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images).to(device)   # batch x
                b_y = Variable(labels).to(device)   # batch y
                embedding = transformer(b_x.float())   
                output = classifier(embedding)         
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()    
                # apply gradients             
                optimizer.step()                
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test(model, classifier):
    # Test the model
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = getCrops(images)
            embedding = model(Variable(images).to(device).float())
            test_output = classifier(embedding)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            labels = Variable(labels).to(device)
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.5f' % accuracy)

def main():
    classifier = Classifier()
    classifier.apply(weights_init)
    classifier.to(device)
    transformer = TransformerEncoder(dim=16,blocks=3,heads=4)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    train(EPOCH, transformer.to(device), loaders, optimizer, classifier)
    torch.save(transformer.state_dict(), './base_trans.pth')
    torch.save(classifier.state_dict(), './base_classifier.pth')
    # classifier.load_state_dict(torch.load('./base_classifier.pth'))
    # transformer.load_state_dict(torch.load('./base_trans.pth'))
    for param in transformer.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False
    test(transformer, classifier)

    # transformer.load_state_dict(torch.load('./base_teacher.pth'))
    # for param in transformer.parameters():
    #     param.requires_grad = False

    # optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    # student = Encoder()
    # student.load_state_dict(torch.load('./student_noactivate.pth'))
    # for param in student.parameters():
    #     param.requires_grad = False
    # student.out = nn.Linear(8 * 7 * 7, 10)
    
    # student.apply(weights_init)
    # student.to(device)
    # train(1, student, loaders, optimizer)
    # test(student)
    print('Done.')

if __name__ == '__main__':
    main()