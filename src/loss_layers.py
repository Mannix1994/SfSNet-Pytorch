# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from torch.nn import Module


class L1LossLayerWt(Module):
    def __init__(self, wt_real, wt_syn):
        # type: (float, float) -> None
        super(L1LossLayerWt, self).__init__()
        self._wt_real = wt_real
        self._wt_syn = wt_syn

    def forward(self, recon, recon_m, label):
        # type (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        :param recon: rec/arec/recon_mask normal/albedo/reconstructed image
        :param recon_m: normal_m/albedo_m ground truth of normal/albedo/image
        :param label: label2/label3/label1 flag for which data is synthetic
        :return: loss, a scalar
        """
        if recon.size() != recon_m.size():
            raise Exception("Inputs must have the same dimension.")
        diff = recon - recon_m

        loss = 0.
        for i in range(label.size()[0]):
            if label[i] > 0:
                wt = self._wt_real
            else:
                wt = self._wt_syn
            loss += wt * torch.sum(torch.abs(diff[i, ...]))

        loss /= recon.size()[0]
        return loss

    def numpy(self, recon, recon_m, label):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
        diff = recon - recon_m
        _sum = 0
        for i in range(recon.shape[0]):
            if label[i] > 0:
                wt = self._wt_real
            else:
                wt = self._wt_syn
            tmp = wt * np.sum(np.abs(diff[i, ...]))
            _sum += tmp
        return _sum / recon.shape[0]


class L2LossLayerWt(Module):
    def __init__(self, wt_real, wt_syn):
        # type: (float, float) -> None
        super(L2LossLayerWt, self).__init__()
        self._wt_real = wt_real
        self._wt_syn = wt_syn

    def forward(self, fc_light, fc_light_gt, label3):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        :param fc_light: fc_light
        :param fc_light_gt:
        :param label3: flag for which data is synthetic
        :return: loss, a scalar
        """
        diff = fc_light - fc_light_gt

        loss = 0
        for i in range(label3.size()[0]):
            if label3[i] > 0:
                wt = self._wt_real
            else:
                wt = self._wt_syn
            tmp = wt * torch.sum(diff[i, ...] ** 2)
            loss = loss + tmp

        loss = loss/label3.size()[0]/2
        return loss

    def numpy(self, fc_light, label, label3):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
        diff = fc_light - label
        _sum = 0
        for i in range(fc_light.shape[0]):
            if label3[i] > 0:
                wt = self._wt_real
            else:
                wt = self._wt_syn
            tmp = wt * np.sum(diff[i, ...]**2)
            _sum += tmp
        return _sum / fc_light.shape[0] / 2
