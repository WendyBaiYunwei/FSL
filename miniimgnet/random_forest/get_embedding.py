from img2vec_pytorch import Img2Vec
from PIL import Image
import os
import glob
import pickle
import numpy as np
import json

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=False)


def mini_imagenet_folders():
    train_folder = './val'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
    ]

    return metatrain_folders

vectors = []
for folder in mini_imagenet_folders():
    image_list = []
    for f in glob.iglob(folder+"/*"):
        image_list.append(Image.open(f))
    vector = img2vec.get_vec(image_list)
    vectors.append(vector)
vectors = np.array(vectors)
print(vectors.shape)
vectors = np.reshape(vectors, (-1, 512))
with open("embedding_val.pkl","wb") as f:
    pickle.dump(vectors, f)