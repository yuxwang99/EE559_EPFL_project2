from mini_framework import *


class Linear_model(Module):
    def __init__(self):
        super(Linear_model, self).__init__()
        self.dim_in, self.dim_out = 2, 2

        self.layers = Sequential(Linear(self.dim_in, 25, bias=False),
                                 # BatchNorm1d(25, affine=True),
                                 Sigmoid(),
                                 Linear(25, 25),
                                 Sigmoid(),
                                 Linear(25, 25),
                                 Sigmoid(),
                                 Linear(25, 25),
                                 Sigmoid(),
                                 Linear(25, self.dim_out),
                                 BatchNorm1d(self.dim_out, affine=True),
                                 Sigmoid(),
                                 )


    def forward(self, data):
        return self.layers.forward(data)

    def backward(self, gradwrtoutput, eta):
        self.layers.backward(gradwrtoutput, eta)

#
