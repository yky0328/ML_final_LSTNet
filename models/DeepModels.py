
import math
import random
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.Attention import *


"""
    Ref: https://github.com/laiguokun/LSTNet
         https://arxiv.org/abs/1703.07015
    Implemented by PyTorch.
"""

class LSTNet(nn.Module):
    def __init__(self, args, data):
        super(LSTNet, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip

        self.is_SSA = 1

        self.pt = (self.P - self.Ck) // self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            if self.is_SSA:
                self.highway = nn.Linear(128 + self.hw, 1)
            else:
                self.highway = nn.Linear(self.hw, 1)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh



        # spatial Attention
        if self.is_SSA:
            self.encoder_f = nn.LSTM(input_size=self.P, hidden_size=128, num_layers=1, batch_first=True)
            self.attention_f = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=2)
            self.alpha = nn.Parameter(torch.Tensor(1), requires_grad=True)



    def forward(self, x):
        batch_size = x.size(0)

        # spatial attention
        if self.is_SSA:
            att2f = torch.tensor(0), torch.tensor(0)
            x_f, _ = self.encoder_f(x.permute(0, 2, 1))
            att2f = self.attention_f(x_f, x_f, x_f)



        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            # spatial attetion
            if self.is_SSA:
                x = x[:, -self.hw:, :]
                x = x.permute(0, 2, 1).contiguous().view(-1, self.hw)
                z = x_f + self.alpha * att2f
                z = z.contiguous().view(-1, 128)
                z = self.highway(torch.cat((x, z), 1))
            else:
                z = x[:, -self.hw:, :]
                z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
                z = self.highway(z)

            z = z.view(-1, self.m)
            res = res + z

        if self.output:
            res = self.output(res)
        return res


