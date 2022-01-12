from torch.utils.data import Dataset
import numpy as np
import  PIL
import os
import torch
from skimage import io, transform
import pickle

class OrderedTG(Dataset):
    def __init__(self, transform):
        dbfile = open('pToDiff.pkl', 'rb')     
        pairToK = pickle.load(dbfile)
        dbfile.close()
        # print(pairToK)
        pairToK = sorted(pairToK.items(), key = lambda x : x[1], reverse=False) ##true 
        pairToK = dict(pairToK)
        self.queryPaths = []
        self.queryYs = []
        self.supportPaths = []
        for querys, queryYs, support in pairToK:
            querys = querys[2:-2].split('\', \'')
            self.queryPaths.append(querys)
            queryYs = queryYs[8:-2].split(', ')
            self.queryYs.append(queryYs)
            support = support[2:-2].split('\', \'')
            self.supportPaths.append(support)
        self.transform = transform

    def __getitem__(self, index):
        queryXs = []
        queryYs = []
        supportXs = []
        for i in range(5): # change if change batch size
            queryX = io.imread(os.path.abspath(self.queryPaths[index][i]))
            queryX = self.transform(queryX)
            queryXs.append(queryX.numpy())
            queryY = self.queryYs[index][i]
            queryYs.append(int(queryY))
            supportX = io.imread(os.path.abspath(self.supportPaths[index][i]))
            supportX = self.transform(supportX)
            supportXs.append(supportX.numpy())
        return torch.FloatTensor(queryXs), torch.LongTensor(queryYs), torch.FloatTensor(supportXs)
        # return self.queryPaths[index], self.queryYs[index], self.supportPaths[index]

    def __len__(self):
        return len(self.queryPaths)