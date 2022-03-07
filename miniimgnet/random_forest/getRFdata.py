# input data
# output an RF labeller class that can train and predict

import numpy as np
import task_generator as tg
import os
import random
import pickle
import json


class RFdata():
    def getData(self, classStart, classEnd, type='Train'):
        diffLevel = 30 #30, 7, 200, 4: 58.96%
        iterations = 7
        Xs = []
        Ys = []
        for classI in range(classStart, classEnd):
            forbiddenIndices = set([i for i in range(classI, classI + 600)])
            fullIndices = set([i for i in range(classStart * 600, classEnd * 600)])
            indices = fullIndices - forbiddenIndices
            for i in range(iterations):
                embeddingI = classI * 600 + i
                cur = np.expand_dims(self.embeddings[embeddingI], axis = 0)
                cur = np.repeat(cur, diffLevel, 0) #20x512
                
                # get sim
                if type == 'Train':
                    simImgs = self.adjList[self.nameList[embeddingI]][:diffLevel]
                else:
                    simImgs = random.sample(self.adjList[self.nameList[embeddingI]], k = diffLevel)
                
                simImgEs = []
                for name in simImgs:
                    idx = self.nameToIdx[name[1]]
                    simImgEs.append(self.embeddings[idx])

                simImgs = np.stack(simImgEs).squeeze()
                concats = np.concatenate([cur, simImgs], axis = 1)
                Xs.append(concats)
                labels = np.repeat(np.ones(1, dtype = np.long), diffLevel, 0)
                Ys.append(labels)

                # get diff
                indices = random.sample(indices, k = diffLevel)
                diffImgs = self.embeddings[indices].squeeze()
                concats = np.concatenate([cur, diffImgs], axis = 1)
                Xs.append(concats)
                labels = np.repeat(np.zeros(1, dtype = np.long), diffLevel, 0)
                Ys.append(labels)

        Xs = np.stack(Xs).reshape(-1, 512 * 2)
        Ys = np.stack(Ys).reshape(-1, 1)

        print(Xs.shape, Ys.shape)
        return Xs, Ys

    def __init__(self):
        with open('embedding_new.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)
        with open('label_list.json', 'r') as f: 
            self.embeddingYs = json.load(f)
        with open('name_list.json', 'r') as f: 
            self.nameList = json.load(f)
        with open('imgNameToIdx.json', 'r') as f: 
            self.nameToIdx = json.load(f)
        with open('embedding_sim.pkl', 'rb') as f: 
            self.adjList = pickle.load(f)

        self.trainClassSize = 50
        self.testClassSize = 64 - self.trainClassSize

        self.trainX, self.trainY = self.getData(0, self.trainClassSize)
        self.testX, self.testY = self.getData(self.trainClassSize, 64, 'test')

        with open('rf_trainX.pkl', 'wb') as f:
            pickle.dump(self.trainX, f)

        with open('rf_trainY.pkl', 'wb') as f:
            pickle.dump(self.trainY, f)

        with open('rf_testX.pkl', 'wb') as f:
            pickle.dump(self.testX, f)

        with open('rf_testY.pkl', 'wb') as f:
            pickle.dump(self.testY, f)

RFdata()
# class RF_Labeller():
#     def __init__(self, data_x, data_y):
        
#         self.x = data_x
#         self.y = data_y

#     def train(self):
#         self.classifier.fit(self.x[:len(self.x) // 4 * 3], self.y[:len(self.x) // 4 * 3])

#     def predict(self):
#         return self.classifier.predict(self.x[len(self.x) // 4 * 3:])

#     def eval(self):
#         preds = self.predict()
#         accuracy = accuracy_score(preds, self.y[len(self.x) // 4 * 3:])
#         print('Accuracy:', accuracy*100, '%.')

# # test RFL
# dataset = MiniImgnet()
# print('done initing dataset')
# rfl = RF_Labeller(dataset.train_x, dataset.train_y)
# print('start training')
# rfl.train()
# print('start testing')
# rfl.eval()
