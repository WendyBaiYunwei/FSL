import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
from self_attention_cv import ResNet50ViT
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("-test","--isTest",type = bool, default=True)
args = parser.parse_args()

torch.manual_seed(0)

isTest = args.isTest
EPOCH = 1
BATCH_SIZE = 1
DIM = 28
DIM2 = 6
studentPth = './trans_student.pth'
classifierPth = './base_classifier.pth'
teacherPth = './base_trans.pth'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=3,           
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 8, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.hidden = nn.Linear(8 * 7 * 7, 4)
        self.out = nn.Linear(4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        return x

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
    loss = torch.square(out - target)
    return loss

def train(trainloader, teacher_trans, student, teacher_cls, optimizer, scheduler, device):
    print("Training...")
    
    student.train()

    for i in range(EPOCH):
        epoch_loss = 0
        count = 0
        for inputs, _ in trainloader:
            sample_features = student(Variable(inputs).to(device))

            teacherIn = inputs.repeat(1, 3, 1, 1)
            baseline_features = teacher_trans(Variable(teacherIn).to(device).float()) # 16 * 32 * 7 * 7
            baseline_features = baseline_features.flatten(start_dim = 1)
            baseline_features = teacher_cls(Variable(baseline_features).to(device)).flatten(start_dim = 1)

            optimizer.zero_grad()

            loss = get_loss(sample_features, baseline_features)

            loss.backward(torch.ones_like(sample_features))

            optimizer.step()

            epoch_loss += torch.sum(torch.sum(loss)).item()
            if count % 1000 == 0:
                print(count, epoch_loss / (count + 1))
            count += 1

        scheduler.step()
        torch.save(student.state_dict(), studentPth)


def test(testloader, student, teacher_cls, device):
    print("Testing...")
    count = 0
    for inputs, labels in testloader:
        sample = student(Variable(inputs).to(device))
        output = teacher_cls(Variable(sample).to(device).float())
        pred_y = torch.max(output, 1)[1].data.squeeze()
        labels = Variable(labels).to(device)
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        count += 1
        if count % 1000 == 0:
            print(count)
    print('Test Accuracy of the model on the 10000 test images: %.5f' % accuracy)

def main():
    device = torch.device("cuda")

    teacher_cls = Classifier()
    teacher_cls.load_state_dict(torch.load(classifierPth))
    for param in teacher_cls.parameters():
        param.requires_grad = False

    student = CNN()
    if isTest == False:
        teacher_cls = teacher_cls.hidden1
        teacher_cls.to(device)
        teacher_trans = ResNet50ViT(img_dim=DIM, pretrained_resnet=True, 
                    blocks=3, classification=False, 
                    dim_linear_block=DIM2, dim=DIM2)
        teacher_trans.load_state_dict(torch.load(teacherPth))
        teacher_trans.to(device)
        for param in teacher_trans.parameters():
            param.requires_grad = False
        
        student.apply(weights_init).to(device)
        optimizer = torch.optim.Adam([
        {"params": student.hidden.parameters(), "lr": 0.001},####0.002
        {"params": student.conv1.parameters(), "lr": 0.01},
        {"params": student.conv2.parameters(), "lr": 0.01},
        ])
        scheduler = StepLR(optimizer,step_size=1,gamma=0.9)
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = transforms.ToTensor(),
            download = False,            
        )

        trainloader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True, 
                                            num_workers=1)
        student.to(device)

        train(trainloader, teacher_trans, student, teacher_cls, optimizer, scheduler, device)

    else:
        teacher_cls = teacher_cls.classifier
        teacher_cls.to(device)
        student.load_state_dict(torch.load(studentPth))
        train_data = datasets.MNIST(
            root = 'data',
            train = False,                         
            transform = transforms.ToTensor(),
            download = False,            
        )

        testloader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True, 
                                            num_workers=1)
        student.to(device)
        test(testloader, student, teacher_cls, device)
        
    print('Done.')

# 600/15, 800
if __name__ == '__main__':
    main()