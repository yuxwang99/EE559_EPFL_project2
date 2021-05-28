from mini_framework import *


class Linear_model(Module):
    """
    Linear model implemented with pytorch
    """
    def __init__(self):
        super(Linear_model, self).__init__()
        self.dim_in, self.dim_out = 2, 2

        self.layers = Sequential(Linear(self.dim_in, 25, bias=False),
                                 # BatchNorm1d(25, affine=True),
                                 Tanh(),
                                 Linear(25, 25),
                                 Tanh(),
                                 Linear(25, 25),
                                 Tanh(),
                                 Linear(25, 25),
                                 Tanh(),
                                 Linear(25, self.dim_out),
                                 BatchNorm1d(self.dim_out, affine=True),
                                 Tanh(),
                                 )


    def forward(self, data):
        return self.layers.forward(data)

    def backward(self, gradwrtoutput, eta):
        self.layers.backward(gradwrtoutput, eta)


class CNN_model(Module):
    """
    CNN model implemented with PyChrot
    """
    def __init__(self):
        super(CNN_model, self).__init__()
        self.dim_in, self.dim_out = 2, 2

        self.layers = Sequential(
                                 Conv1D(in_channel=1, out_channel=1, kernel_size=1, stride=1),
                                 Flatten(),
                                 ReLU(),
                                 Linear(2, 25),
                                 ReLU(),
                                 Linear(25, 25),
                                 ReLU(),
                                 Linear(25, 25),
                                 Sigmoid(),
                                 Linear(25, 2),
                                 Sigmoid(),
                                 )


    def forward(self, data):
        data = data.unsqueeze(dim=1)
        return self.layers.forward(data)

    def backward(self, gradwrtoutput, eta):
        self.layers.backward(gradwrtoutput, eta)
