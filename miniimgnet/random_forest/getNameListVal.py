import pickle
import numpy as np
import json
import os

train_roots = []
test_roots = []

test_folder = './val'

class_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

for c in class_folders:
    temp = [x for x in os.listdir(c)]
    train_roots.extend(temp)

with open('name_list_val.json', 'w') as f:
    json.dump(train_roots, f)