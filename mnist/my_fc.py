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
        res = Variable(torch.ones((len(x), self.h_in * self.w_in, self.size_channel))).to(self.device)
        for n in range(len(x)):
            one_x = torch.transpose(x[n], 0, 1) #512 * 7 * 7
            one_x = torch.transpose(one_x, 1, 2)
            one_x = torch.flatten(one_x, end_dim = 1) #7 * 7 * 512
            x_times_w = Variable(torch.ones((self.h_in * self.w_in, self.size_channel))).to(self.device)
            #x_times_w = torch.ones((self.h_in * self.w_in, self.size_channel))
            for i in range(self.size_channel):   
                new_x = one_x[:, i].view((7, 7))
                new_weights = self.weight[:, i].view((7, 7))
                x_times_w[:, i] = torch.add(torch.flatten(torch.mm(new_x, new_weights)), self.bias[:, i])
            res[n] = x_times_w
        return res