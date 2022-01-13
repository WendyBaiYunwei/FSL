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
        # self.pairs = []
        # for querys, queryYs, support in pairToK:
        #     querysL = querys[2:-2].split('\', \'')
        #     queryYsL = queryYs[8:-2].split(', ')
        #     supportL = support[2:-2].split('\', \'')
        #     loss = pairToK[(querys, queryYs, support)]
        #     for i in range(len(querysL)):
        #         e = [querysL[i], int(queryYsL[i]), supportL, loss[i]]
        #         self.pairs.append(e)
        # ttlLen = len(self.pairs)
        # self.pairs = sorted(self.pairs[:ttlLen//3*2], key = lambda x : x[3], reverse=True) ##true sort
        # self.pairs = sorted(self.pairs, key = lambda x : x[3], reverse=True) ##true sort
        self.pairToK = sorted(pairToK.items(), key = lambda x : x[1])
        self.transform = transform

    def __getitem__(self, index):
        # queryX = io.imread(os.path.abspath(self.pairs[index][0]))
        # queryX = self.transform(queryX)
        # queryY = self.pairs[index][1]
        # supportXs = []
        # for i in range(5): # change if change batch size
        #     supportX = io.imread(os.path.abspath(self.pairs[index][2][i]))
        #     supportX = sxsupportX)
        #     supportXs.append(supportX.numpy())
        # supportXs = np.array(supportXs)
        print(self.pairToK[index])
        for queryXs, queryYs, supportXs in self.pairToK[index]:
            queryXl = queryXs[2:-2].split('\', \'')
            queryYl = queryYs[8:-2].split(', ')
            supportXl = supportXs[2:-2].split('\', \'')
            queryXs = []
            queryYs = []
            supportXs = []
            for i in range(5):
                x = queryXl[i]
                x = io.imread(os.path.abspath(x))
                x = self.transform(x)
                y = queryYl[i]
                s = supportXl[i]
                s = io.imread(os.path.abspath(s))
                s = self.transform(s)

                queryXs.append(x)
                queryYs.append(y)
                supportXs.append(s)

        return  torch.from_numpy(queryXs.values), \
            torch.from_numpy(queryYs.values), torch.from_numpy(supportXs.values)

    def __len__(self):
        return len(self.pairToK)