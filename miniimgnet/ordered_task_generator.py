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
        self.pairs = []
        for querys, queryYs, support in pairToK:
            querysL = querys[2:-2].split('\', \'')
            queryYsL = queryYs[8:-2].split(', ')
            supportL = support[2:-2].split('\', \'')
            loss = pairToK[(querys, queryYs, support)]
            for i in range(len(querysL)):
                e = [querysL[i], int(queryYsL[i]), supportL, loss[i]]
                self.pairs.append(e)
        # ttlLen = len(self.pairs)
        # pairs = sorted(self.pairs[:ttlLen//3*2], key = lambda x : x[3], reverse=True) ##true sort
        # self.pairs = sorted(self.pairs, key = lambda x : x[3], reverse=True) ##true sort
        self.transform = transform

    def __getitem__(self, index):
        queryX = io.imread(os.path.abspath(self.pairs[index][0]))
        queryX = self.transform(queryX)
        queryY = self.pairs[index][1]
        supportXs = []
        for i in range(5): # change if change batch size
            supportX = io.imread(os.path.abspath(self.pairs[index][2][i]))
            supportX = self.transform(supportX)
            supportXs.append(supportX.numpy())
        supportXs = np.array(supportXs)
        return queryX, queryY, torch.from_numpy(supportXs)

    def __len__(self):
        return len(self.pairs)