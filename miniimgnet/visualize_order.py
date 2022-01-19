import pickle
import torch
import cv2
import matplotlib.pyplot as plt
import os

dbfile = open('pToDiff_sorted.pkl', 'rb')
pairToK = pickle.load(dbfile)
dbfile.close()

# pairToK = sorted(pairToK.items(), key = lambda x : x[1], reverse=True)

# pToDiff = dict(pairToK)
# with open('pToDiff_sorted.pkl', 'wb') as out:
#     pickle.dump(pToDiff, out)
ptkList = list(pairToK.items())

TAKE = 3
l = ptkList[:TAKE]
l.extend(ptkList[-TAKE:])

supportNames = []
queryNames = []

for i in range(len(l)):
    queryNames.extend(l[i][0][0][2:-2].split('\', \''))
    supportNames.extend(l[i][0][2][2:-2].split('\', \''))

pairs = []
for i in range(len(supportNames)):
    pairs.append((supportNames[i], queryNames[i]))

k = 0
for supportNames, queryNames in pairs:
    if k % 5 == 0:
        fig, axs = plt.subplots(5, 2)
    support = cv2.imread(supportNames)
    axs[k % 5, 0].imshow(support)
    query = cv2.imread(queryNames)
    axs[k % 5, 1].imshow(query)
    

    if k % 5 == 4:
        plt.savefig(str(k) + '.png')
    
    k += 1