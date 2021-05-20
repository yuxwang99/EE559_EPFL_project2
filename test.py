from mini_framework import F_MSE, MSE, L2loss, L1loss
from models import Linear_model
from models_pt import Linear_model_pt
import random
import time
import torch
import math
import logger
import os
import shutil

from helper import *





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

    model = Linear_model()
    batch_size = 10
    n_epoch = 200
    losses = []
    step_size = 1 # Sigmoid 1 , ReLU 0.001

    train_data, test_data, train_label, test_label = build_data()
    train_label_oh, test_label_oh = to_one_hot(train_label), to_one_hot(test_label)

    mse = MSE()
    all_loss=[]
    for i in range(n_epoch):
        total_loss = 0
        num_batch = len(train_data.split(batch_size))
        for batch_data, label in zip(train_data.split(batch_size), train_label_oh.split(batch_size)):
            y = model.forward(batch_data)
            # print(y)
            loss = mse.forward(y, label)
            total_loss += loss
            all_loss.append(loss)
            dloss = mse.backward()
            model.backward(dloss, step_size)

        y_train = model.forward(train_data)
        _, result = torch.max(y_train, 1)

        if i % 20 == 0:
            p, r, f = metric(result, train_label)
            logger.info("\n"
                        f'The train precision of {i}th epoch is {p}\n'
                        f'The train recall    of {i}th epoch is {r}\n'
                        f'The train f1        of {i}th epoch is {f}\n')
            y_test = model.forward(test_data)
            _, result = torch.max(y_test, 1)
            p, r, f = metric(result, test_label)
            logger.info("\n"
                        f'The test precision of {i}th epoch is {p}\n'
                        f'The test recall    of {i}th epoch is {r}\n'
                        f'The test f1        of {i}th epoch is {f}\n')
            logger.info(f'currant loss per data point is  {total_loss/num_batch}')
            if i != 0:
                if i < 200 & i % 100 == 0:
                    step_size = step_size / 2
                elif i > 200 & i % 50 == 0:
                    step_size = step_size / 2
        losses.append(total_loss)

