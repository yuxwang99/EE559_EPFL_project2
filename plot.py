import numpy as np
import matplotlib.pyplot as plt
from helper import build_data


def plot_data():
    train_data, _, train_label, _ = build_data()

    train_data = train_data.numpy()
    train_label = train_label.numpy()
    label1 = train_label == 1
    label0 = train_label == 0
    train_data1 = train_data[label1]
    train_label1 = train_label[label1]
    train_data0 = train_data[label0]
    train_label0 = train_label[label0]
    plt.figure(figsize=(6, 6), dpi=80)

    inside = plt.scatter(x=train_data1[:, 0], y=train_data1[:, 1], c="r", alpha=0.5, s =1.5)
    outside = plt.scatter(x=train_data0[:, 0], y=train_data0[:, 1], c="b", alpha=0.5, s=1.5)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.legend((inside, outside), ("interior point","exterior point"))
    plt.xlim([-0.25, 1.25])
    plt.ylim([-0.25, 1.25])
    plt.title("train data distribution")
    plt.savefig("plots/data_scatter.png")

def plot_loss():
    raise NotImplementedError

if __name__ == '__main__':
    plot_data()