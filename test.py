from mini_framework import*
import random
import torch
import math
def to_one_hot(classes):
    n = classes.size()[0]
    d = max(classes)+1
    classes_oh = torch.zeros(n,d.int())
    classes_oh[range(n),classes.long()] = 1
    return classes_oh

train_data, test_data = torch.rand([1000,2]), torch.rand([1000,2])

d_train, d_test = torch.sqrt((train_data[:,0]-0.5)**2+(train_data[:,1]-0.5)**2),\
                    torch.sqrt((test_data[:,0]-0.5)**2+(test_data[:,1]-0.5)**2)

train_label, test_label = torch.zeros(1000),torch.zeros(1000)
train_label[d_train<1/torch.sqrt(2*torch.tensor(math.pi))] = 1
test_label[d_test<1/torch.sqrt(2*torch.tensor(math.pi))] = 1

train_label_oh, test_label_oh = to_one_hot(train_label), to_one_hot(test_label)

class Linear_model(Module):
    def __init__(self):
        super(Linear_model, self).__init__()
        self.dim_in, self.dim_out = 2, 2

        self.layers = Sequential(Linear(self.dim_in, 25),
                                  Tanh(),
                                  Linear(25, 25),
                                  Tanh(),
                                  Linear(25, 25),
                                  Tanh(),
                                  Linear(25, 25),
                                  Tanh(),
                                  Linear(25, self.dim_out),
                                  Tanh())

    def forward(self, data):
        return self.layers.forward(data)

    def backward(self, label, y, eta):
        self.layers.backward(label, y, eta)

if __name__ == '__main__':

    model = Linear_model()
    batch_size = 10
    n_epoch = 1000
    losses = []
    step_size = 0.8/batch_size
    for i in range(n_epoch):
        total_loss = 0
        for batch_data, label in zip(train_data.split(batch_size), train_label_oh.split(batch_size)):
            y = model.forward(batch_data)
            loss = MSELoss(y,label)
            total_loss += loss
            model.backward(label, y, step_size)

        y = model.forward(train_data)
        _, result = torch.max(y, 1)

        if i%20 == 0:
            print('The accurancy of', i, 'th epoch is ', sum(result == train_label) / 1000.0)
            print('currant loss is', total_loss)
            if i != 0:
                if i<200 & i%100==0:
                    step_size = step_size / 2
                elif i>200 & i%50==0:
                    step_size = step_size / 2
        losses.append(total_loss)
        # new_idx = torch.randperm(1000)
        # train_data = train_data[new_idx,:]
        # train_label_oh = train_label_oh[new_idx,:]

    y = model.forward(test_data)
    _, result = torch.max(y, 1)
    print('The accurancy of test data is ', sum(result == test_label) / 1000.0)


