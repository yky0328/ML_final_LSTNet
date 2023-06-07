import random
import torch
import numpy as np
from torch.autograd import Variable
import pickle as pkl
import pandas as pd


def load_data(file, type):
    """
    load data
    :param file: input file (T * N)
    :param type: Type of input file, including 'txt', 'npy', 'other'(pickle type), *'Tensor'
    :return: a 2-dim array [T, N]
    """
    
    if type == 'txt' or type == 'csv':
        data = np.loadtxt(file, delimiter=',')
    else:
        with open(file, 'rb') as f:
            data = pkl.load(f)
    return data

def gen_seq(data, n_frame):
    """
    Generate data in the form of standard sequence unit.
    :param data: 2-d np.array, [T, N], input time series.
    :param n_frame: int, Size of slide window.
    :return: np.array, [n_unit, n_frame, n_series]
    """

    n_unit = data.shape[0] - n_frame + 1
    n_series = data.shape[1]
    seq = np.zeros((n_unit, n_frame, n_series))
    for i in range(n_unit):
        sta = i
        end = sta + n_frame
        seq[i, :, :] = np.reshape(data[sta:end, :], [n_frame, n_series])
    return seq


def gen_batch(inputs, batch_size, shuffle=True):
    """
    Data iterator in batch.
    :param inputs: np.array, [n_unit, n_frame, n_series, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param shuffle: bool, whether shuffle the batches.
    :return: batch...
    """
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            end_idx = len_inputs
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

#
class Data_utility(object):
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda
        self.P = window
        self.h = horizon
        d = np.asarray(load_data(file_name, file_name[-3:]))
        self.rawdat = d[:500,:]
        print(self.rawdat.shape)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        if self.cuda:
            # self.scale = self.scale.cuda()
            self.scale = self.scale.to('mps')
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))


    def _normalized(self, normalize):
        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)

        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                if self.scale[i] == 0:
                    self.dat[:, i] = self.rawdat[:, i]
                else:
                    self.dat[:, i] = self.rawdat[:, i] / self.scale[i]

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train, _ = self._batchify(train_set, self.h)
        self.valid, _ = self._batchify(valid_set, self.h)
        self.test, self.test_m = self._batchify(test_set, self.h)


    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        Y_m = torch.zeros((n, self.h, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
            Y_m[i, :, :] = torch.from_numpy(self.dat[end:idx_set[i]+1, :])
        return [X, Y], Y_m

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                # X = X.cuda()
                # Y = Y.cuda()
                X = X.to('mps')
                Y = Y.to('mps')
            yield Variable(X), Variable(Y)
            start_idx += batch_size
