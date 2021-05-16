from mini_framework import F_MSELoss
from models import Linear_model
from models_pt import Linear_model_pt
import random
import time
import torch
import math
import logger
import os
import shutil

def to_one_hot(classes):
    n = classes.size()[0]
    d = max(classes) + 1
    # TODO why d.int but classes.long ? maybe both should be long ?
    classes_oh = torch.zeros(n, d.int())
    classes_oh[range(n), classes.long()] = 1
    return classes_oh


def build_data():
    train_data, test_data = torch.rand([1000, 2]), torch.rand([1000, 2])

    d_train, d_test = torch.sqrt((train_data[:, 0] - 0.5) ** 2 + (train_data[:, 1] - 0.5) ** 2), \
                      torch.sqrt((test_data[:, 0] - 0.5) ** 2 + (test_data[:, 1] - 0.5) ** 2)

    train_label, test_label = torch.zeros(1000), torch.zeros(1000)
    train_label[d_train < 1 / torch.sqrt(2 * torch.tensor(math.pi))] = 1
    test_label[d_test < 1 / torch.sqrt(2 * torch.tensor(math.pi))] = 1



    return train_data, test_data, train_label, test_label





if __name__ == '__main__':

    # set up logs
    log_dir = "logs"
    time_str = time.strftime('%m-%d-%H-%M')
    log_name = "{}".format(time_str)
    log_dir = os.path.join(log_dir, log_name)
    logger.set_logger_dir(log_dir)

    code_dir = os.path.dirname(__file__) or "."
    os.mkdir(os.path.join(log_dir, "code"))
    for f in os.scandir(code_dir):
        if f.is_file() and f.name.endswith(".py"):
            shutil.copy(f.path, os.path.join(log_dir, "code"))



    model = Linear_model_pt()
    logger.info(model)
    # logger.info(model.parameters())

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"name={name}, param={param.data}")
    batch_size = 10
    n_epoch = 1000
    losses = []
    step_size = 10 / batch_size
    opt = torch.optim.SGD(model.parameters(), lr=step_size, momentum=.0)

    train_data, test_data, train_label, test_label = build_data()
    train_label_oh, test_label_oh = to_one_hot(train_label), to_one_hot(test_label)
    for i in range(n_epoch):
        total_loss = 0
        for batch_data, label in zip(train_data.split(batch_size), train_label_oh.split(batch_size)):
            opt.zero_grad()
            y = model(batch_data)
            mseloss = torch.nn.MSELoss()
            loss = mseloss(y, label)
            total_loss += loss.float()
            loss.backward()
            opt.step()

        y = model.forward(train_data)
        _, result = torch.max(y, 1)

        if i % 20 == 0:
            logger.info(f'The accurancy of {i}th epoch is {sum(result == train_label) / 1000.0}')
            logger.info(f'currant loss is  {total_loss}')
            if i != 0:
                if i < 200 & i % 100 == 0:
                    step_size = step_size / 2
                elif i > 200 & i % 50 == 0:
                    step_size = step_size / 2
        losses.append(total_loss)
        # new_idx = torch.randperm(1000)
        # train_data = train_data[new_idx,:]
        # train_label_oh = train_label_oh[new_idx,:]

    y = model.forward(test_data)
    _, result = torch.max(y, 1)
    logger.info(f'The accurancy of test data is { sum(result == test_label) / 1000.0}')
