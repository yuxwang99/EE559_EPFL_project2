from unittest import TestCase

import torch
import numpy as np

from mini_framework import *

class TestTanh(TestCase):
    def setUp(self):
        self.Tanh = Tanh()
    def test_forward(self):
        """
        Tanh should return 1 even for large values
        torch.Float will fail at 100
        torch.Double will fail at 900
        """
        large_value = torch.ones(1)*10000
        small_value = torch.ones(1) * -10000
        res = self.Tanh.forward(large_value)
        self.assertFalse(torch.isnan(res))

        """
        tanh should have the same value as torch.tnah
        """
        self.assertEqual(torch.tanh(large_value),self.Tanh.forward(large_value))

        self.assertEqual(torch.tanh(small_value), self.Tanh.forward(small_value))

    def test_backward(self):
        self.fail()
