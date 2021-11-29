import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from self_attention_cv import ResNet50ViT
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


model = RandomForestRegressor(max_depth=200, random_state=0)
model.fit(X, y)
preds = model.predict(X_test)
print(np.mean(mean_squared_error(y_test, preds)))
# init RF
# prepare data
# load rf, get MSEloss
# test rf