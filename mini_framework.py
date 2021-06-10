from torch import empty, Generator
import math
from collections import OrderedDict

# fix random seed for parameter initialization
g_cpu = Generator()
g = g_cpu.manual_seed(2147483646)

def F_MSE(pred, target):
    """
    Calculate MSE loss of a prediction given the ground true values
    Args:
        pred: predicted value
        target: ground true value

    Returns: MSE loss

    """
    n = len(target)
    return F_L2(pred, target)/n


def F_L2(pred, target):
    """
    Calculate L2 loss of a prediction given the ground true values
    Args:
        pred: predicted value
        target: ground true value

    Returns: L2 loss
    """
    dif = (pred - target).view(-1, 1)
    return dif.pow(2).sum()


def F_L1(pred, target):
    """
    Calculate L1 loss of a prediction given the ground true values
    Args:
        pred: predicted value
        target: ground true value

    Returns: L1 loss
    """
    dif = (pred - target).view(-1, 1)
    return dif.abs().sum()


class Module(object):
    """ base class of all Neural Net modules"""
    def __init__(self):
        self._module = OrderedDict()
        self._cache = OrderedDict()  # save the output results for each layers

    def __call__(self, *input):
        return self.forward(*input)

    def forward(self, *input):
        """
        forward pass
        Args:
            *input: take the ouput from the previous layer as input

        Returns:
            output will be send to next layer as its input
        """
        raise NotImplementedError


    def backward(self, *gradwrtoutput):
        """
        backward pass to calculate gradient
        Args:
            *gradwrtoutput:  gradient wrt the output variables of this layer in the forward pass

        Returns:
            gradient wrt the input variables of this layer in the forward pass
        """
        raise NotImplementedError

    def param(self):
        """
        return a list  parameters and its gradient  of a given Neural Net Module
        Returns: a list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        This list is empty for parameterless modules (e.g. ReLU).

        """
        has_weight = hasattr(self, 'weight') and self.weight is not None
        has_bias = hasattr(self, 'bias') and self.bias is not None
        if has_weight and has_bias:
            return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]
        elif has_weight:
            return [(self.weight, self.weight_grad)]
        elif has_bias:
            return [(self.bias, self.bias_grad)]
        else:
            return




class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def add_module(self, name: str, module):
        self._module[name] = module

    def add_cache(self, name: str, result):
        self._cache[name] = result

    def clear_cache(self):
        self._cache = OrderedDict()

    def forward(self, data):
        cache = data
        for module in self._module:
            data, cache = self._module[module].forward(data)
            self.add_cache(module, cache)
        return data

    def backward(self, gradwrtoutput, eta):
        # dl_dout = dloss(y, label)
        dl_dout = gradwrtoutput
        for idx in range(len(self._module))[::-1]:
            idx = str(idx)
            dl_dout = self._module[idx].backward(dl_dout, self._cache[idx], eta)
        self.clear_cache()


class MSE(Module):
    """
    Module to measure the mean square error (MSE) between each element in
    the input
    """
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, pred, target):
        output = F_MSE(pred, target)
        self.cache = (pred - target)
        return output

    def backward(self):
        return 2*self.cache/self.cache.size(0)


class L2loss(Module):
    """
    Module to measure the L2 error between each element in
    the input
    """
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, pred, target):
        output = F_L2(pred, target)
        self.cache = (pred - target)
        return output

    def backward(self):
        return 2*self.cache


class L1loss(Module):
    """
    Module to measure the L1 error between each element in
    the input
    """
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, pred, target):
        output = F_L1(pred, target)
        self.cache = (pred - target)
        return output

    def backward(self):
        self.cache[self.cache < 0] = -1
        self.cache[self.cache >= 0] = 1
        return self.cache



class Tanh(Module):
    """Applies the HardTanh function element-wise"""
    def forward(self, data):
        output = 1-2/((2*data.float()).exp()+1)
        cache = data
        return output, cache

    def backward(self, dl, x, eta):
        return dl * (1 - self.forward(x)[0].pow(2))


class BatchNorm1d(Module):
    """
    Applies Batch Normalization over an input tensor, if affine = True, a linear transformation is applied
    """
    expected_dim = 2
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()

        self.affine = affine
        self.eps = eps
        self.running_mean = empty(num_features)
        self.running_std = empty(num_features)

        if self.affine:
            self.weight = empty(num_features)
            self.bias = empty(num_features)
            self.weight_grad = empty(num_features)
            self.bias_grad = empty(num_features)

        self.reset_parameters()

    def forward(self, x):
        N, D = x.shape
        mu = 1. / N * x.sum(dim=0)
        xmu = x - mu
        sq = xmu ** 2
        var = 1. / N * sq.sum(dim=0)
        sqrtvar = (var + self.eps).sqrt()
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        cache = (xhat, xmu, ivar, sqrtvar, var)

        if not self.affine:
            return xhat, cache

        gammax = self.weight * xhat
        out = gammax + self.bias
        return out, cache

    def backward(self, dl, cache, eta):
        # unfold the variables stored in cache
        xhat, xmu, ivar, sqrtvar, var = cache
        N, D = dl.shape

        if self.affine:
            dbeta = dl.sum(dim=0)
            dgamma = (dl * xhat).sum(dim=0)
            dxhat = dl * self.weight

            self.weight_grad = eta * dgamma
            self.bias_grad = eta * dbeta
            self.weight = self.weight - self.weight_grad
            self.bias = self.bias - self.bias_grad

        else:
            dxhat = dl

        divar = (dxhat * xmu).sum(dim=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. / (sqrtvar ** 2) * divar
        dvar = 0.5 * 1. / (var + self.eps).sqrt() * dsqrtvar
        dsq = 1. / N * empty((N, D)).new_ones((N, D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * (dxmu1 + dxmu2).sum(dim=0)
        dx2 = 1. / N * empty((N, D)).new_ones((N, D)) * dmu
        dx = dx1 + dx2
        return dx

    def check_input_dim(self,input):
        if input.dim()!= self.expected_dim:
            raise RuntimeError('only mini-batch supported ({}D tensor), got {}D tensor instead'.format(self.expected_dim, input.dim()))
        if input.size(1) != self.running_mean.size():
            raise RuntimeError('got {}-feature tensor, expected {}'.format(input.size(1), self.running_mean.size()))

    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_(generator=g)
            self.bias.data.zero_()

class ReLU(Module):
    """Applies the rectified linear unit function element-wise"""
    def forward(self, data):
        output = data*(data > 0)
        cache = data
        return output, cache

    def backward(self, dl, x, eta):
        x[x < 0] = 0
        x[x >= 0] = 1
        return dl * x


class Sigmoid(Module):
    """Applies the sigmoid function element-wise """
    def forward(self, data):
        output = 1/((-data.float()).exp()+1)
        cache = data
        return output, cache

    def backward(self, dl, x, eta):

        return dl*self.forward(x)[0]*(1-self.forward(x)[0])


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b
    """
    def __init__(self, dim_in, dim_out, bias: bool=True):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = 0
        self.use_bias = bias
        self.weight = empty([dim_in, dim_out])
        self.weight_grad = empty([dim_in, dim_out])
        if self.use_bias:
            self.bias = empty([dim_out])
            self.bias_grad = empty([dim_out])
        self.reset_parameters()

    def forward(self, x):
        self.batch_size = x.size(0)
        y = x @ self.weight
        if self.use_bias:
            y += self.bias
        cache = x
        return y, cache

    def backward(self, dl, cache, eta=5 * 1e-1):
        self.weight_grad = cache.t() @ dl
        self.weight = self.weight - eta * self.weight_grad
        if self.use_bias:
            self.bias_grad = dl.sum(axis=0)
            self.bias = self.bias - eta * self.bias_grad
        return dl @ (self.weight + eta * self.weight_grad).t()

    def reset_parameters(self):
        stdv = math.sqrt(6/(self.weight.size()[0]+self.weight.size()[1]))
        self.weight.data.uniform_(-stdv, stdv, generator=g)
        if self.use_bias:
            self.bias.data.uniform_(-stdv, stdv, generator=g)


class Conv1D(Module):
    """Applies a 1D convolution over an input signal composed of several input
    planes."""
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Conv1D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.kernel = empty([out_channel, in_channel, kernel_size])
        self.kernel_grad = empty([out_channel, in_channel, kernel_size])
        self.bias = empty([out_channel])
        self.bias_grad = empty([out_channel])
        self.reset_parameters()

    def forward(self, x):
        N, C_in, L_in = x.size()
        assert C_in == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)

        L_out = int((L_in - self.kernel_size) / self.stride) + 1
        Z = empty([N, self.out_channel, L_out])

        for c in range(self.out_channel):
            for l in range(L_out):
                start = self.stride * l
                end = start + self.kernel_size
                x_slice = x[:, :, start: end]
                Z[:, c, l]= self.conv_single_step(x_slice, self.kernel[c], self.bias[c])
        cache = {"x": x}
        return Z, cache

    def backward(self, dl, cache, eta=5 * 1e-1):
        x = cache["x"]
        N, C_in, L_in = x.size()
        dx = empty([N, C_in, L_in])*0.
        _, C_out, L_out = dl.size()

        for n in range(N):
            for c_out in range(self.out_channel):
                for l in range(L_out):
                    start = self.stride * l
                    end = start + self.kernel_size
                    x_slice = x[n, :, start: end]
                    dx[n, :, start: end] += self.kernel[c_out] * dl[n, c_out, l]
                    self.kernel_grad[c_out] += x_slice * dl[n, c_out, l]

        self.bias_grad = dl.sum([0, 2])
        
        self.kernel = self.kernel - eta * self.kernel_grad
        self.bias = self.bias - eta * self.bias_grad
        
        return dx

    def reset_parameters(self):
        stdv = math.sqrt(6/(self.in_channel+self.out_channel))
        self.kernel.data.uniform_(-stdv, stdv, generator=g)
        self.bias.data.uniform_(-stdv, stdv, generator=g)

    def conv_single_step(self, input, W, b):
        return (input*W+b).sum([1, 2])


class Flatten(Module):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x_0 = x.shape[0]
        self.x_1 = x.shape[1]
        self.x_2 = x.shape[2]
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2]),None

    def backward(self, x, cache, eta=5 * 1e-1):
        return x.reshape(self.x_0, self.x_1, self.x_2)
