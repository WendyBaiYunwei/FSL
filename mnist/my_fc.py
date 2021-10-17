import torch
import torch.nn as nn
from torch.autograd import Variable

class MyLinearLayer(nn.Module):
    def __init__(self, h_in, w_in, size_channel):
        super().__init__()
        self.h_in, self.w_in, self.size_channel = h_in, w_in, size_channel
        weight = torch.Tensor(h_in * w_in, size_channel)
        self.weight = nn.Parameter(weight)
        bias = torch.Tensor(h_in * w_in, size_channel)
        self.bias = nn.Parameter(bias)
        self.device = torch.device("cuda")

    #N * 512 * 7 * 7 -> 49, 512
    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        x = torch.flatten(x, end_dim = 2)
        x_times_w = Variable(torch.ones((self.h_in * self.w_in, self.size_channel))).to(self.device)
        for i in range(self.size_channel):   
            new_x = x[:, i].view((7, 7))
            new_weights = self.weight[:, i].view((7, 7))
            x_times_w[:, i] = torch.add(torch.flatten(torch.mm(new_x, new_weights)), self.bias[:, i])
        return x_times_w