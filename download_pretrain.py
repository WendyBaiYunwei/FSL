import torch 
import torchvision
import os

os.environ['TORCH_HOME'] = './vgg16-pretrained' #setting the environment variable
vgg16 = torchvision.models.vgg16(pretrained=True)