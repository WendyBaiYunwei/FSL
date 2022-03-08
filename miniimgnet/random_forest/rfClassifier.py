from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
import numpy as np
import torch

# relation pairs: {class1query1...class1query15,class2query1...}
# rf relations are rounded to the nearest 0.01
# rfRelations = rf.getBatchRelScores(supportNames, batchQueryNames) #(relation_pairs_sizex1) -> embedding as additional channel

class RF():
    def __init__(self):
        print('loading dataset')
        with open('rf_trainX.pkl', 'rb') as f:
            trainX = pickle.load(f)

        with open('rf_trainY.pkl', 'rb') as f:
            trainY = pickle.load(f)

        with open('embedding_new.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)

        with open('embedding_val.pkl', 'rb') as f:
            self.embeddingsVal = pickle.load(f)

        with open('imgNameToIdx.json', 'r') as f:
            self.nameToIdx = json.load(f)

        with open('imgNameToIdxVal.json', 'r') as f:
            self.nameToIdxVal = json.load(f)

        print(trainX.shape, trainY.shape)
        print('start RF training')
        self.classifier = RandomForestRegressor(n_estimators = 200, random_state = 42, max_features=4)
        self.classifier.fit(trainX, trainY) #to-do
        del trainX
        del trainY

    def getBatchRelScores(self, supportNames, batchQueryNames):
        relations = []
        for sName in supportNames:
            sName = sName[len('./train/n03347037/'):]
            sEmbedding = self.embeddings[self.nameToIdx[sName]]
            for qName in batchQueryNames:
                qName = qName[len('./train/n03347037/'):]
                qEmbedding = self.embeddings[self.nameToIdx[qName]]
                concat = np.concatenate([sEmbedding, qEmbedding], axis = 1).squeeze()
                relations.append(concat)
        relations = np.stack(relations).round(2)
        # print(relations)
        # print(relations.shape)
        preds = self.classifier.predict(relations)
        preds *= 100
        preds = preds.astype(int)
        return torch.from_numpy(preds).cuda()#to-do: check rf is correct

    def getBatchRelScoresVal(self, supportNames, batchQueryNames):
        relations = []
        for sName in supportNames:
            sName = sName[len('./val/n03347037/'):]
            sEmbedding = self.embeddingsVal[self.nameToIdxVal[sName]]
            for qName in batchQueryNames:
                qName = qName[len('./val/n03347037/'):]
                qEmbedding = self.embeddingsVal[self.nameToIdxVal[qName]]
                concat = np.concatenate([sEmbedding, qEmbedding], axis = 1).squeeze()
                relations.append(concat)
        relations = np.stack(relations).round(2)
        # print(relations)
        # print(relations.shape)
        preds = self.classifier.predict(relations)
        preds *= 100
        preds = preds.astype(int)
        return torch.from_numpy(preds).cuda()
