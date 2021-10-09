import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
import copy
import numpy as np
import os

def main():
    batch_size = 32
    device = torch.device("cuda")
 
    vgg16 = models.vgg16(pretrained=True)

    vgg16.to(device)
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./cifar-data', train=True,
                                    download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./cifar-data', train=False,
                                    download=True, transform=transform)

    num_ftrs = vgg16.fc.in_features
    vgg16.fc = nn.Linear(num_ftrs, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in trainloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes
                epoch_acc = running_corrects.double() / dataset_sizes
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                # deep copy the model
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    correct = 0
    total = 0
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    print("Done")

if __name__ == '__main__':
    main()
