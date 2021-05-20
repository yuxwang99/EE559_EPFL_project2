# from models_pt import Linear_model_pt
#
# import time
# import logger
# import os
# import shutil
#
# from helper import *
#
#
# if __name__ == '__main__':
#
#     # set up logs
#     log_dir = "logs"
#     time_str = time.strftime('%m-%d-%H-%M')
#     log_name = "{}_pt".format(time_str)
#     log_dir = os.path.join(log_dir, log_name)
#     logger.set_logger_dir(log_dir)
#
#     code_dir = os.path.dirname(__file__) or "."
#     os.mkdir(os.path.join(log_dir, "code"))
#     for f in os.scandir(code_dir):
#         if f.is_file() and f.name.endswith(".py"):
#             shutil.copy(f.path, os.path.join(log_dir, "code"))
#
#     model = Linear_model_pt()
#     mseloss = torch.nn.MSELoss()
#
#     batch_size = 10
#     n_epoch = 200
#     losses = []
#     step_size = 1
#     opt = torch.optim.SGD(model.parameters(), lr=step_size, momentum=.0)
#
#     train_data, test_data, train_label, test_label = build_data()
#     train_label_oh, test_label_oh = to_one_hot(train_label), to_one_hot(test_label)
#     for i in range(n_epoch):
#         total_loss = 0
#         num_batch = len(train_data.split(batch_size))
#         for batch_data, label in zip(train_data.split(batch_size), train_label_oh.split(batch_size)):
#             opt.zero_grad()
#             y = model(batch_data)
#
#             loss = mseloss(y, label)
#             total_loss += loss.float()*2 #
#             loss.backward()
#             opt.step()
#
#         y = model.forward(train_data)
#         _, result = torch.max(y, 1)
#
#         if i % 20 == 0:
#             p, r, f = metric(result, train_label)
#             logger.info("\n"
#                         f'The train precision of {i}th epoch is {p}\n'
#                         f'The train recall    of {i}th epoch is {r}\n'
#                         f'The train f1        of {i}th epoch is {f}\n')
#             y_test = model.forward(test_data)
#             _, result = torch.max(y_test, 1)
#             p, r, f = metric(result, test_label)
#             logger.info("\n"
#                         f'The test precision of {i}th epoch is {p}\n'
#                         f'The test recall    of {i}th epoch is {r}\n'
#                         f'The test f1        of {i}th epoch is {f}\n')
#             logger.info(f'currant loss per data point is  {total_loss/num_batch}')
#             if i != 0:
#                 if i < 200 & i % 100 == 0:
#                     step_size = step_size / 2
#                 elif i > 200 & i % 50 == 0:
#                     step_size = step_size / 2
#         losses.append(total_loss)
