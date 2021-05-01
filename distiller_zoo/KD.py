from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, use_softmax=True):
        super(DistillKL, self).__init__()
        self.T = T
        self.use_softmax = use_softmax

    def forward(self, y_s, y_t):
        if self.use_softmax:
            p_s = F.log_softmax(y_s/self.T, dim=1)
            p_t = F.softmax(y_t/self.T, dim=1)
        else:   
            p_s = y_s ** (1 / self.T)
            p_t = y_t ** (1 / self.T)
            p_s = p_s / torch.sum(p_s, dim=1).unsqueeze(-1)
            p_t = p_t / torch.sum(p_t, dim=1).unsqueeze(-1)
            p_s = torch.log(p_s)
            print('****')
        # print(p_t)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        # print(loss)
        return loss
