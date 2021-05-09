from torch import empty
import torch
import logger
import math
from collections import OrderedDict

x = empty([11, 2])


def MSELoss(v, t):
    dif = (v - t).view(-1, 1)
    losses = torch.sum(torch.pow(dif, 2))
    return torch.sqrt(losses)


def dloss(v, t):
    return (v - t) / MSELoss(v, t)


class Module(object):
    def __init__(self):
        self._module = OrderedDict()
        self._cache = OrderedDict()  # save the output results for each layers

    def forward(self, *input): raise NotImplementedError

    def backward(self, *gradwrtoutput): raise NotImplementedError

    def param(self): return []

    def add_module(self, name: str, module):
        self._module[name] = module

    def add_cache(self, name: str, result):
        self._cache[name] = result

    def clear_cache(self):
        self._cache = OrderedDict()


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, data):
        cache = data
        for module in self._module:
            data, cache = self._module[module].forward(data)
            self.add_cache(module, cache)
        return data

    def backward(self, label, y, eta):
        dl_dout = dloss(y, label)
        for idx in range(len(self._module))[::-1]:
            # idx = str(len(self._module) - int(module_idx) - 1)
            idx = str(idx)
            dl_dout = self._module[idx].backward(dl_dout, self._cache[idx], eta)
        self.clear_cache()


class Tanh(Module):
    def forward(self, data):
        # TODO This implementation will cause nan, if data is a large value , such as 100

        # num = torch.exp(data.double()) - torch.exp(-data.double())
        # den = torch.exp(data.double()) + torch.exp(-data.double())
        output = 1-2/(torch.exp(2*data.float())+1)
        cache = data
        return output, cache

    def backward(self, dl, x, eta):
        return dl * (1 - torch.pow(self.forward(x)[0], 2))


class BatchNorm(Module):
    expected_dim = 2
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()

        self.affine = affine
        self.eps = eps
        self.running_mean = empty(num_features)
        self.running_std = empty(num_features)

        if self.affine:
            self.gamma = empty(num_features)
            self.beta = empty(num_features)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.gamma.data.uniform_()
            self.beta.data.zero_()

        # self.running_mean.zero_()
        # self.running_var.fill_(1)

    def forward(self, x):
        N, D = x.shape

        # step1: calculate mean
        mu = 1. / N * torch.sum(x, dim=0)

        # step2: subtract mean vector of every trainings example
        xmu = x - mu

        # step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step4: calculate variance
        var = 1. / N * torch.sum(sq, dim=0)

        # step5: add eps for numerical stability, then sqrt
        sqrtvar = torch.sqrt(var + self.eps)

        # step6: invert sqrtwar
        ivar = 1. / sqrtvar

        # step7: execute normalization
        xhat = xmu * ivar

        cache = (xhat, xmu, ivar, sqrtvar, var)

        if not self.affine:
            return xhat, cache

        # step8: the two transformation steps
        gammax = self.gamma * xhat

        # step9
        out = gammax + self.beta

        # store intermediate
        return out, cache

    def backward(self, dl, cache, eta):
        # unfold the variables stored in cache
        xhat, xmu, ivar, sqrtvar, var = cache

        # get the dimensions of the input/output
        N, D = dl.shape

        if self.affine:
            # step9
            dbeta = torch.sum(dl, dim=0)
            dgammax = dl  # not necessary, but more understandable

            # step8
            dgamma = torch.sum(dgammax * xhat, dim=0)
            dxhat = dgammax * self.gamma

            self.gamma = self.gamma - eta * dgamma
            self.beta = self.beta - eta * dbeta

        else:
            dxhat = dl

        # step7
        divar = torch.sum(dxhat * xmu, dim=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / torch.sqrt(var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * torch.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * torch.sum(dxmu1 + dxmu2, dim=0)

        # step1
        dx2 = 1. / N * torch.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2
        return dx
        # return dl * (torch.std(x, dim=0), 2)


    def check_input_dim(self,input):
        if input.dim()!= self.expected_dim:
            raise RuntimeError('only mini-batch supported ({}D tensor), got {}D tensor instead'.format(self.expected_dim, input.dim()))
        if input.size(1) != self.running_mean.size():
            raise RuntimeError('got {}-feature tensor, expected {}'.format(input.size(1), self.running_mean.size()))


class ReLU(Module):
    def forward(self, data):
        output = data*(data > 0)
        cache = data
        return output, cache

    def backward(self, dl, x, eta):
        x[x < 0] = 0
        x[x >= 0] = 1
        return dl * x


class Sigmoid(Module):
    #TODO not working yet
    def forward(self, data):
        output = 1/(torch.exp(-data.float())+1)
        cache = data
        return output, cache

    def backward(self, dl, x, eta):

        return dl*self.forward(x)[0]*(1-self.forward(x)[0])



class Linear(Module):
    #TODO 没有加bias
    def __init__(self, dim_in, dim_out, bias: bool=True):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = 0
        self.use_bias = bias
        # self.parameters = empty([dim_in, dim_out]).normal_(0, 1 / (3 * self.dim_out))
        self.weight = empty([dim_in, dim_out])
        if self.use_bias:
            self.bias = empty([dim_out])
        self.reset_parameters()

    def forward(self, x):
        self.batch_size = x.size(0)
        y = x @ self.weight
        if self.use_bias:
            y += self.bias
        cache = x
        return y, cache

    def backward(self, dl, data, eta=5 * 1e-1):
        dl_dw = data.t() @ dl
        self.weight = self.weight - eta * dl_dw
        if self.use_bias:
            dl_db = dl.sum(axis=0)
            self.bias = self.bias - eta * dl_db
        return dl @ (self.weight + eta * dl_dw).t()


    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size()[0])
        self.weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            self.bias.data.uniform_(-stdv, stdv)

