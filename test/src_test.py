from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import unittest
from torch import from_numpy as fn

from src import L1LossLayerWt, L2LossLayerWt, ShadingLayer, NormLayer, ChangeFormLayer


class LossLayersTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.l1_loss = L1LossLayerWt(0.6, 0.4)
        self.l2_loss = L2LossLayerWt(0.6, 0.4)
        self.data = np.random.randn(16, 3, 128, 128).astype(dtype=np.float32)
        self.ground_truth = np.random.randn(16, 3, 128, 128).astype(dtype=np.float32)
        self.label = np.random.randint(0, 2, (16,)).astype(dtype=np.float32)
        self.fc_light = np.random.randn(16, 27).astype(dtype=np.float32)
        self.fc_light_ground_truth = np.random.randn(16, 27).astype(dtype=np.float32)

    def testL1LossLayerWtCPU(self):
        n_out = self.l1_loss.numpy(self.data, self.ground_truth, self.label)
        # test cpu
        t_out = self.l1_loss(fn(self.data), fn(self.ground_truth), fn(self.label))
        print(np.sum(np.abs(t_out.numpy() - n_out)))
        self.assertTrue(np.sum(np.abs(t_out.numpy() - n_out)) < 1e-3)

    def testL1LossLayerWtGPU(self):
        if torch.cuda.is_available():
            n_out = self.l1_loss.numpy(self.data, self.ground_truth, self.label)
            # test gpu
            t_out = self.l1_loss(fn(self.data).cuda(), fn(self.ground_truth).cuda(), fn(self.label).cuda())
            self.assertTrue(np.sum(np.abs(t_out.cpu().numpy() - n_out)) < 1e-2)

    def testL2LossLayerWtCPU(self):
        n_out = self.l2_loss.numpy(self.fc_light, self.fc_light_ground_truth, self.label)
        # test cpu
        t_out = self.l2_loss(fn(self.fc_light), fn(self.fc_light_ground_truth), fn(self.label))
        self.assertTrue(np.sum(np.abs(t_out.numpy() - n_out)) < 1e-5)

    def testL2LossLayerWtGPU(self):
        if torch.cuda.is_available():
            n_out = self.l2_loss.numpy(self.fc_light, self.fc_light_ground_truth, self.label)
            # test gpu
            t_out = self.l2_loss(fn(self.fc_light).cuda(), fn(self.fc_light_ground_truth).cuda(), fn(self.label).cuda())
            self.assertTrue(np.sum(np.abs(t_out.cpu().numpy() - n_out)) < 1e-5)


if __name__ == '__main__':
    unittest.main()
