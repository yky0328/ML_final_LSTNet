
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


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel=5, dropout=0.5, hidden_size=1, INN=True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = 1
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1  # by default: stride==1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1  # by default: stride==1

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1  # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),

            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes=in_planes, splitting=True,
                                kernel=kernel, dropout=dropout, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)


class LevelSCINet(nn.Module):
    def __init__(self, in_planes, kernel_size, dropout, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes=in_planes, kernel=kernel_size, dropout=dropout,
                                        hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)  # even: B, T, D odd: B, T, D


class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, hidden_size, INN):
        super().__init__()
        self.current_level = current_level

        self.workingblock = LevelSCINet(
            in_planes=in_planes,
            kernel_size=kernel_size,
            dropout=dropout,
            hidden_size=hidden_size,
            INN=INN)

        if current_level != 0:
            self.SCINet_Tree_odd = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, hidden_size,
                                               INN)
            self.SCINet_Tree_even = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, hidden_size,
                                                INN)

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure.
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))


class EncoderTree(nn.Module):
    def __init__(self, in_planes, num_levels, kernel_size,  hidden_size, INN, dropout=0.5):
        super().__init__()
        self.levels = num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes=in_planes,
            current_level=num_levels - 1,
            kernel_size=kernel_size,
            dropout=dropout,
            hidden_size=hidden_size,
            INN=INN)

    def forward(self, x):
        x = self.SCINet_Tree(x)

        return x

class LSTNet(nn.Module):
    def __init__(self, args, data, modified=True):
        super(LSTNet, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip

        # sci block
        self.input_dim = args.input_dim
        self.sci_kernel_size = args.sci_kernel_size
        self.hidden_size = args.sci_hidden_size

        # self-attention
        self.is_SSA = 0
        self.is_TSA = 0

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

        # sci block
        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels=3,
            kernel_size=self.sci_kernel_size,
            hidden_size=self.hidden_size,
            INN=modified)

        # spatial self-attention
        if self.is_SSA:
            self.encoder_f = nn.LSTM(input_size=self.P, hidden_size=128, num_layers=1, batch_first=True)
            self.attention_f = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=2)
            self.alpha = nn.Parameter(torch.Tensor(1), requires_grad=True)

        # temporal self-attention
        if self.is_TSA:
            # self.attention_t = ScaledDotProductAttention(d_model=self.m, d_k=self.m, d_v=self.m, h=6)
            # self.belta = nn.Parameter(torch.Tensor(1), requires_grad=True)

            # self.attention_r = ScaledDotProductAttention(d_model=100, d_k=100, d_v=100, h=1)
            # self.gamma_r = nn.Parameter(torch.Tensor(1), requires_grad=True)

            self.attention_g = ScaledDotProductAttention(d_model=self.hidR, d_k=self.hidR, d_v=self.hidR, h=1)
            self.gamma_g = nn.Parameter(torch.Tensor(1), requires_grad=True)

            # self.attention_l = ScaledDotProductAttention(d_model=45, d_k=45, d_v=45, h=1)
            # self.gamma_f = nn.Parameter(torch.Tensor(1), requires_grad=True)


    def forward(self, x):
        batch_size = x.size(0)

        # spatial attention
        if self.is_SSA:
            att2f = torch.tensor(0), torch.tensor(0)
            x_f, _ = self.encoder_f(x.permute(0, 2, 1))
            att2f = self.attention_f(x_f, x_f, x_f)

        # temporal attention （for a batch ???）
        # if self.is_TSA:
        #     att2t = self.attention_t(x, x, x)
        #     x = x + self.belta * att2t

        # sci block
        res1 = self.blocks1(x)
        x = x + res1
        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        # Attention for global window
        if self.is_TSA:
            att2g = self.attention_g(r, r, r)
            r = r + self.gamma_g * att2g

        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)

            # Attention for local skipped window
            # if self.is_TSA:
            #     att2g = self.attention_l(s.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
            #     s = s.unsqueeze(1) + self.gamma_f * att2g
            #     s = s.squeeze(1)

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


