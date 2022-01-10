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
        pairToK = sorted(pairToK.items(), key = lambda x : x[1], reverse=True) 
        pairToK = dict(pairToK)
        self.queryPaths = []
        self.queryYs = []
        self.allSupportPaths = []
        for query, queryY, support in pairToK:
            self.queryPaths.append(query)
            self.queryYs.append(queryY)
            self.allSupportPaths.append(support)
        self.transform = transform

    def __getitem__(self, index):
        queryX = io.imread(os.path.abspath(self.queryPaths[index]))
        queryX = self.transform(queryX)
        queryY = self.queryYs[index]
        supportXs = []
        supportPaths = self.allSupportPaths[index][2:-2].split('\', \'')
        for supportX in supportPaths:
            supportX = io.imread(os.path.abspath(supportX))
            supportX = self.transform(supportX)
            supportXs.append(supportX.numpy())
        return queryX, queryY, torch.FloatTensor(supportXs)

    def __len__(self):
        return len(self.queryPaths)