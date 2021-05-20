from unittest import TestCase
from helper import *


class Test(TestCase):
    def test_recall(self):
        true_label = [1, 0, 1, 1]
        pred_label = [1, 0, 1, 0]
        r = recall(pred_label, true_label, label=1)
        self.assertEqual(2 / 3, r)

    def test_precision(self):
        true_label = [1, 0, 1, 1]
        pred_label = [1, 0, 1, 0]
        r = precision(pred_label, true_label, label=1)
        self.assertEqual(1, r)
