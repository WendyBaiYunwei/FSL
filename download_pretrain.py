import torch 
import torchvision
import os

# Suppose you are trying to load pre-trained resnet model in directory- models\resnet

os.environ['TORCH_HOME'] = './vgg16-pretrained' #setting the environment variable
vgg16 = torchvision.models.vgg16(pretrained=True)