from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
torch.manual_seed(0)

def train(data):
    labels = torch.load('labels.pt')


if __name__ == '__main__':
    trainData = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transforms.ToTensor(),
        download = False,            
    )

    trainloader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=1)
    train(trainData)
    test(testData)