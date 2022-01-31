# input data
# output an RF labeller class that can train and predict

from sklearn.ensemble import RandomForestClassifier
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import random
from sklearn.metrics import accuracy_score


class MiniImgnet():
    def __init__(self):
        metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

        def get_raw_x_y(folder, size):
            labels = np.array(range(len(folder)))
            labels = dict(zip(folder, labels))
            self.train_roots = []
            for c in folder:
                temp = [os.path.join(c, x) for x in os.listdir(c)]
                self.train_roots.extend(temp)

            random.seed(0)
            random.shuffle(self.train_roots)

            train_roots = self.train_roots[:size]
            train_labels = [labels[self.get_class(x)] for x in train_roots]

            return train_roots, train_labels

        def transform(x, y):
            xs = []
            ys = []
            for index in range(len(x)):
                x_path = os.path.abspath('./' + x[index])
                input = io.imread(x_path)
                input = np.swapaxes(input, 0, 2)
                input = np.swapaxes(input, 1, 2)
                label = int(y[index])
                xs.append(input)
                ys.append(label)
            return np.stack(xs), np.stack(ys)

        TRAIN_SIZE =3600
        TEST_SIZE = 50

        train_roots, train_labels = get_raw_x_y(metatrain_folders, TRAIN_SIZE)
        test_roots, test_labels = get_raw_x_y(metatest_folders, TEST_SIZE)

        self.train_x, self.train_y = transform(train_roots, train_labels)
        self.test_x, self.test_y = transform(test_roots, test_labels)

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        x = torch.tensor(x, dtype=float)
        out = self.layer1(x.float())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1)
        return out # 64

class RF_Labeller():
    def __init__(self, data_x, data_y, x_test, y_test):
        self.encoder = CNNEncoder()
        self.classifier = RandomForestClassifier(n_estimators = 12, random_state = 42, max_features=4)
        self.x = data_x
        self.y = data_y
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        x = self.encoder(self.x).detach().numpy()
        self.classifier.fit(x, self.y)

    def predict(self):
        x = self.encoder(self.x_test).detach().numpy()
        return self.classifier.predict(x)

    def eval(self):
        preds = self.predict()
        accuracy = accuracy_score(preds , self.y_test)
        print('Accuracy:', accuracy*100, '%.')

# test RFL
dataset = MiniImgnet()
print('done initing dataset')
rfl = RF_Labeller(dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y)
print('start training')
rfl.train()
print('start testing')
rfl.eval()
