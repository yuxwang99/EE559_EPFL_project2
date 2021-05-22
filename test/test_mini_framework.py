from unittest import TestCase

import torch
from mini_framework import *
from torch.autograd import Variable
import numpy as np
from helper import *
from torch import nn
np.random.seed(0)

class TestTanh(TestCase):
    def setUp(self):
        self.Tanh = Tanh()

    def test_forward(self):
        """
        Tanh should return 1 even for large values
        torch.Float will fail at 100
        torch.Double will fail at 900
        """
        large_value = torch.ones(1) * 10000
        small_value = torch.ones(1) * -10000
        res = self.Tanh.forward(large_value)
        self.assertFalse(torch.isnan(res))

        """
        tanh should have the same value as torch.tnah
        """
        self.assertEqual(torch.tanh(large_value), self.Tanh.forward(large_value))

        self.assertEqual(torch.tanh(small_value), self.Tanh.forward(small_value))

    def test_backward(self):
        self.fail()


class TestLinear(TestCase):
    def setUp(self):
        self.linear = Linear(12, 16)

    def test_parameters(self):
        self.linear = Linear(12, 16)
        self.assertEqual(2, len(self.linear.param()))

    def test_parameters_no_bias(self):
        self.linear = Linear(12, 16, bias=False)
        self.assertEqual(1, len(self.linear.param()))

    def test_forward(self):
        self.linear.forward()

    def test_backward(self):
        self.fail()


class TestConv1D(TestCase):
    def test_forward(self):

        N = 8
        C = 1
        L = 24
        x = torch.ones([N, C, L]) * 0.2
        conv1d = Conv1D(in_channel=1, out_channel=3, kernel_size=3, stride=1)
        y = conv1d.forward(x)

        m = torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=2)
        output = m(x)

        print(output)

    def test_end2end(self):
        ## initialize your layer and PyTorch layer
        net1 = Conv1D_ref(8, 12, 3, 2)
        net2 = torch.nn.Conv1d(8, 12, 3, 2)
        net3 = Conv1D(8, 12, 3, 2)
        ## initialize the inputs
        x1 = np.random.rand(3, 8, 20)
        x2 = Variable(torch.tensor(x1), requires_grad=True)
        x3 = x2
        ## Copy the parameters from the Conv1D class to PyTorch layer
        net2.weight = nn.Parameter(torch.tensor(net1.W))
        net2.bias = nn.Parameter(torch.tensor(net1.b))
        net3.kernel = torch.tensor(net1.W)
        net3.bias = torch.tensor(net1.b)
        ## Your forward and backward
        y1 = net1(x1)
        b, c, w = y1.shape
        delta = np.random.randn(b, c, w)
        dx = net1.backward(delta)
        ## PyTorch forward and backward
        y2 = net2(x2)
        delta = torch.tensor(delta)
        y2.backward(delta)
        ## Your forward and backward
        y3, cache = net3(x3)
        b, c, w = y3.shape
        delta = torch.tensor(np.random.randn(b, c, w))
        dx = net3.backward(delta, cache)

        ## Compare
        def compare(x, y):
            y = y.detach().numpy()
            print(abs(x).max())
            return

        compare(y1, y2)
        compare(dx, x2.grad)
        compare(net1.dW, net2.weight.grad)
        compare(net1.db, net2.bias.grad)


class Test(TestCase):
    def test_f_mse(self):
        data2, data1, label2, label1 = build_data()
        train_label_oh, test_label_oh = to_one_hot(label2), to_one_hot(label1)
        mseloss = torch.nn.MSELoss(reduction="mean")
        loss_pt = mseloss(data2, data1)*data2.size()[1]
        mse = MSE()
        loss = mse(data2, data1)
        print(loss_pt)
        print(loss)

        self.assertEqual(loss_pt, loss)
