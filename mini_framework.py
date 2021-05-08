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
        self._result = OrderedDict()  # save the output results for each layers

    def add_module(self, name: str, module):
        self._module[name] = module

    def add_result(self, name: str, result):
        self._result[name] = result

    def clear_result(self):
        self._result = OrderedDict()


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, data):
        for module in self._module:
            self.add_result(module, data)
            data = self._module[module].forward(data)
        return data

    def backward(self, label, y, eta):
        dl_dout = dloss(y, label)
        for idx in range(len(self._module))[::-1]:
            # idx = str(len(self._module) - int(module_idx) - 1)
            idx = str(idx)
            dl_dout = self._module[idx].backward(dl_dout, self._result[idx], eta)
        self.clear_result()


class Tanh(Module):
    def forward(self, data):
        # TODO This implementation will cause nan, if data is a large value , such as 100

        # num = torch.exp(data.double()) - torch.exp(-data.double())
        # den = torch.exp(data.double()) + torch.exp(-data.double())

        return 1-2/(torch.exp(2*data.float())+1)

    def backward(self, dl, x, eta):
        return dl * (1 - torch.pow(self.forward(x), 2))

    def param(self):
        return []

class Linear(Module):
    #TODO 没有加bias
    def __init__(self, dim_in, dim_out, bias: bool=True):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = 0
        # self.parameters = empty([dim_in, dim_out]).normal_(0, 1 / (3 * self.dim_out))
        self.weight = empty([dim_in, dim_out])
        self.bias = empty([dim_out])
        self.reset_parameters()

    def forward(self, x):
        self.batch_size = x.size(0)
        y = x @ self.weight+self.bias

        return y

    def backward(self, dl, data, eta=5 * 1e-1):
        dl_dw = data.t() @ dl
        dl_db = dl
        # logger.info(f"gradient = { eta * gradwrtoutput}")
        self.weight = self.weight - eta * dl_dw
        self.bias = self.bias - eta * dl_db.sum(axis=0)
        return dl @ (self.weight + eta * dl_dw).t()

    # def param(self):
    #     return self.parameters

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size()[0])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

