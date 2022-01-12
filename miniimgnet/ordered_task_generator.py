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
        pairs = sorted(self.pairs, key = lambda x : x[3], reverse=True) ##true sort
        # pairs.extend(self.pairs[ttlLen//3*2:])
        self.batches = []

        for i in range(0, len(pairs), 5):
            queryXs = []
            queryYs = []
            for queryX, queryY, _, _ in list(pairs[i : i+5]):
                queryXs.append(queryX)
                queryYs.append(queryY)
            l = [queryXs, queryYs, pairs[i][2]]
            self.batches.append(l)

        self.transform = transform

    def __getitem__(self, index):
        queryXs = []
        queryYs = []
        supportXs = []
        for i in range(5): # change if change batch size
            queryX = io.imread(os.path.abspath(self.batches[index][0][i]))
            queryX = self.transform(queryX)
            queryXs.append(queryX.numpy())
            queryY = self.batches[index][1][i]
            queryYs.append(int(queryY))
            supportX = io.imread(os.path.abspath(self.batches[index][2][i]))
            supportX = self.transform(supportX)
            supportXs.append(supportX.numpy())
        return torch.FloatTensor(queryXs), torch.LongTensor(queryYs), torch.FloatTensor(supportXs)

    def __len__(self):
        return len(self.batches)