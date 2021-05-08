from torch.nn import *


class Linear_model_pt(Module):
    def __init__(self):
        super(Linear_model_pt, self).__init__()
        self.dim_in, self.dim_out = 2, 2

        self.layers = Sequential(Linear(self.dim_in, 25, bias=False),
                                 Sigmoid(),
                                 # Linear(25, 25),
                                 # ReLU(),
                                 # Linear(25, 25),
                                 # ReLU(),
                                 # Linear(25, 25),
                                 # ReLU(),
                                 Linear(25, self.dim_out, bias=False),
                                 Sigmoid()
                                 )

    def forward(self, data):
        return self.layers.forward(data)

    def backward(self, label, y, eta):
        self.layers.backward(label, y, eta)