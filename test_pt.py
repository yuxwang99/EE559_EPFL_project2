from mini_framework import F_MSE
from models import Linear_model
from models_pt import Linear_model_pt
import random
import time
import torch
import math
import logger
import os
import shutil

from helper import to_one_hot,build_data


if __name__ == '__main__':

    # set up logs
    log_dir = "logs"
    time_str = time.strftime('%m-%d-%H-%M')
    log_name = "{}_pt".format(time_str)
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


    y = model.forward(test_data)
    _, result = torch.max(y, 1)
    logger.info(f'The accurancy of test data is { sum(result == test_label) / 1000.0}')
