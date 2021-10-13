import torch
import torch.nn as nn
import numpy as np

class MyLinearLayer(nn.Module):
    def __init__(self, h_in, w_in, size_channel):
        super().__init__()
        self.h_in, self.w_in, self.size_channel = h_in, w_in, size_channel
        weight = torch.Tensor(h_in * w_in, size_channel)
        self.weight = nn.Parameter(weight)
        bias = torch.Tensor(h_in * w_in, size_channel)
        self.bias = nn.Parameter(bias)

    #N * 512 * 7 * 7
    def forward(self, x):
        x = np.swapaxes(x, 1, 3)
        x = torch.flatten(x, end_dim = 2)
        x_times_w = torch.ones((self.h_in * self.w_in, self.size_channel))
        for i in range(self.size_channel):    
            x = torch.reshape(x[:, i], (7, 7))
            weights = torch.reshape(self.weight[:, i], (7, 7))
            w = torch.flatten(torch.mm(x, weights))
            x_times_w[:, i] = torch.add(w, self.bias[:, i])
        return x_times_w