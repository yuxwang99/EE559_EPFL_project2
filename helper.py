import torch
import math

torch.manual_seed(0)


def to_one_hot(classes):
    n = classes.size()[0]
    d = max(classes) + 1
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


def precision(pred, true, label):
    """
    calculate prediction precision score of one label
    """
    pred_index = set([ind if pred == label else None for ind, pred in enumerate(pred)])
    pred_index.discard(None)
    true_index = set([ind if pred == label else None for ind, pred in enumerate(true)])
    true_index.discard(None)
    intersect = pred_index.intersection(true_index)

    return len(intersect) / len(pred_index) if len(pred_index) != 0 else 0


def recall(pred, true, label):
    """
    calculate prediction recall score of one label
    """
    pred_index = set([ind if pred == label else None for ind, pred in enumerate(pred)])
    pred_index.discard(None)
    true_index = set([ind if pred == label else None for ind, pred in enumerate(true)])
    true_index.discard(None)
    intersect = pred_index.intersection(true_index)

    return len(intersect) / len(true_index)


def f1(pred, true, label):
    """
    calculate prediction f1 score of one label
    """
    r = recall(pred, true, label)
    p = precision(pred, true, label)

    return 2 * p * r / (p + r) if (p + r) != 0 else 0


def metric_per_label(pred, true, label):
    """
    calculate recall, precision, f1 score of one label in the prediction
    """
    r = recall(pred, true, label)
    p = precision(pred, true, label)
    f = f1(pred, true, label)

    return p, r, f


def metric(pred, true):
    """
    calculate recall, precision, f1 score of each label in the prediction and then take the average
    """
    r_per_label = []
    p_per_label = []
    f_per_label = []
    unique_labels = set(list(true.numpy()))
    for label in unique_labels:
        r = recall(pred, true, label)
        p = precision(pred, true, label)
        f = f1(pred, true, label)
        r_per_label.append(r)
        p_per_label.append(p)
        f_per_label.append(f)
    macro_p = sum(p_per_label) / len(p_per_label)
    macro_r = sum(r_per_label) / len(r_per_label)
    macro_f = sum(f_per_label) / len(f_per_label)
    return macro_p, macro_r, macro_f
