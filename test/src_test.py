from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import unittest
from torch import from_numpy as fn

from src import L1LossLayerWt, L2LossLayerWt, ShadingLayer, NormLayer, ChangeFormLayer


def diff(tensor, array):
    # type: (torch.Tensor, np.ndarray) -> float
    return float(np.sum(np.abs(tensor.cpu().numpy() - array)))


class LossLayersTest(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(LossLayersTest, self).__init__(method_name)
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
        self.assertLess(diff(t_out, n_out), 1e-2)

    def testL1LossLayerWtGPU(self):
        if torch.cuda.is_available():
            n_out = self.l1_loss.numpy(self.data, self.ground_truth, self.label)
            # test gpu
            t_out = self.l1_loss(fn(self.data).cuda(), fn(self.ground_truth).cuda(), fn(self.label).cuda())
            self.assertLess(diff(t_out, n_out), 1e-2)

    def testL2LossLayerWtCPU(self):
        n_out = self.l2_loss.numpy(self.fc_light, self.fc_light_ground_truth, self.label)
        # test cpu
        t_out = self.l2_loss(fn(self.fc_light), fn(self.fc_light_ground_truth), fn(self.label))
        self.assertLess(diff(t_out, n_out), 1e-5)

    def testL2LossLayerWtGPU(self):
        if torch.cuda.is_available():
            n_out = self.l2_loss.numpy(self.fc_light, self.fc_light_ground_truth, self.label)
            # test gpu
            t_out = self.l2_loss(fn(self.fc_light).cuda(), fn(self.fc_light_ground_truth).cuda(), fn(self.label).cuda())
            self.assertLess(diff(t_out, n_out), 1e-5)


class ToolLayersTest(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(ToolLayersTest, self).__init__(method_name)
        self.data = np.random.randn(32, 3, 128, 128)*100
        self.t_data = torch.from_numpy(self.data)
        self.nl = NormLayer()
        self.cfm = ChangeFormLayer()

    def testNormLayerCPU(self):
        t_out = self.nl(self.t_data)
        n_out = self.nl.numpy(self.data)
        self.assertLess(diff(t_out, n_out), 1e-10)

    def testNormLayerGPU(self):
        if torch.cuda.is_available():
            t_out = self.nl(self.t_data.cuda())
            n_out = self.nl.numpy(self.data)
            self.assertLess(diff(t_out, n_out), 1e-5)

    def testChangeFormLayerCPU(self):
        t_out = self.cfm(self.t_data)
        n_out = self.cfm.numpy(self.data)
        self.assertLess(diff(t_out, n_out), 1e-5)

    def testChangeFormLayerGPU(self):
        if torch.cuda.is_available():
            t_out = self.cfm(self.t_data.cuda())
            n_out = self.cfm.numpy(self.data)
            self.assertLess(diff(t_out, n_out), 1e-5)

    def testShadingLayerCPU(self):
        # test cpu
        sl = ShadingLayer(False)
        normal = np.random.randn(16, 3, 128, 128).astype(np.float32)
        fc_lig = np.random.randn(16, 27).astype(np.float32)
        t_out = sl(torch.from_numpy(normal), torch.from_numpy(fc_lig))
        n_out = sl.numpy(normal, fc_lig)
        self.assertLess(diff(t_out, n_out), 1e-5)

    def testShadingLayerGPU(self):
        if torch.cuda.is_available():
            sl = ShadingLayer(True)
            normal = np.random.randn(16, 3, 128, 128).astype(np.float32)
            fc_lig = np.random.randn(16, 27).astype(np.float32)
            t_out = sl(torch.from_numpy(normal).cuda(), torch.from_numpy(fc_lig).cuda())
            n_out = sl.numpy(normal, fc_lig)
            self.assertLess(diff(t_out, n_out), 1e-5)

    def testNormLayer(self):
        t_out = self.nl(self.t_data)
        n_out = self.nl.numpy(self.data)

        # form SfSNet_test.py
        n_out2 = 2 * self.data - 1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2 ** 2, axis=1, keepdims=True)) + 1e-8
        n_out2 = n_out2 / np.repeat(nr, 3, axis=1)

        self.assertLess(np.sum(np.abs(n_out - n_out2)), 1e-10)
        self.assertLess(diff(t_out, n_out2), 1e-10)


if __name__ == '__main__':
    unittest.main()
