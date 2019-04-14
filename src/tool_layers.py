# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from torch.nn import Module


class ChangeFormLayer(Module):
    def __init__(self):
        super(ChangeFormLayer, self).__init__()

    def forward(self, bottom):
        # type: (torch.Tensor) -> torch.Tensor
        return bottom[:, [2, 1, 0], :, :]

    def numpy(self, bottom):
        # type: (np.ndarray) -> np.ndarray
        """
        for verify
        """
        return bottom[:, [2, 1, 0], :, :]


class NormLayer(Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, bottom):
        # type: (torch.Tensor) -> torch.Tensor
        sz = bottom.size()
        nor = bottom * 2 - 1
        ssq = torch.norm(nor, dim=1)
        ssq = ssq.view(sz[0], 1, sz[2], sz[3])
        norm = ssq.repeat(1, sz[1], 1, 1)
        return nor / norm

    def numpy(self, bottom):
        # type: (np.ndarray) -> np.ndarray
        """
        for verify
        """
        sz = bottom.shape
        nor = 2 * bottom - 1
        ssq = np.linalg.norm(nor, axis=1)
        ssq = np.reshape(ssq, (sz[0], 1, sz[2], sz[3]))
        norm = np.tile(ssq, (1, sz[1], 1, 1))
        return np.divide(nor, norm)


class ShadingLayer(Module):
    def __init__(self, gpu):
        super(ShadingLayer, self).__init__()
        self.__gpu = gpu
        self.att = np.pi * np.array([1, 2.0 / 3, 0.25], dtype=np.float32)
        self.att = torch.from_numpy(self.att)
        if self.__gpu:
            self.att = self.att.cuda()
        self.c1 = self.att[0] * (1.0 / np.sqrt(4 * np.pi))
        self.c2 = self.att[1] * (np.sqrt(3.0 / (4 * np.pi)))
        self.c3 = self.att[2] * 0.5 * (np.sqrt(5.0 / (4 * np.pi)))
        self.c4 = self.att[2] * (3.0 * (np.sqrt(5.0 / (12 * np.pi))))
        self.c5 = self.att[2] * (3.0 * (np.sqrt(5.0 / (48 * np.pi))))

    def forward(self, recnormalch, fc_light):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        sz = recnormalch.size()
        top = torch.zeros_like(recnormalch)
        for i in range(0, sz[0]):
            nx = recnormalch[i, 0, ...]
            ny = recnormalch[i, 1, ...]
            nz = recnormalch[i, 2, ...]
            if self.__gpu:
                H1 = self.c1 * torch.ones((sz[2], sz[3]), dtype=torch.float32).cuda()
            else:
                H1 = self.c1 * torch.ones((sz[2], sz[3]), dtype=torch.float32)
            H2 = self.c2 * nz
            H3 = self.c2 * nx
            H4 = self.c2 * ny
            H5 = self.c3 * (2 * nz * nz - nx * nx - ny * ny)
            H6 = self.c4 * nx * nz
            H7 = self.c4 * ny * nz
            H8 = self.c5 * (nx * nx - ny * ny)
            H9 = self.c4 * nx * ny

            for j in range(0, 3):
                L = fc_light[i, j * 9:(j + 1) * 9]
                top[i, j, :, :] = L[0] * H1 + L[1] * H2 + L[2] * H3 + L[3] * H4 \
                                  + L[4] * H5 + L[5] * H6 + L[6] * H7 + L[7] * H8 + L[8] * H9

        return top

    def numpy(self, recnormalch, fc_light):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        sz = recnormalch.shape
        top = np.zeros_like(recnormalch)
        for i in range(0, sz[0]):
            nx = recnormalch[i, 0, ...]
            ny = recnormalch[i, 1, ...]
            nz = recnormalch[i, 2, ...]

            H1 = self.c1.cpu().numpy() * np.ones((sz[2], sz[3]), dtype=np.float32)
            H2 = self.c2.cpu().numpy() * nz
            H3 = self.c2.cpu().numpy() * nx
            H4 = self.c2.cpu().numpy() * ny
            H5 = self.c3.cpu().numpy() * (2 * nz * nz - nx * nx - ny * ny)
            H6 = self.c4.cpu().numpy() * nx * nz
            H7 = self.c4.cpu().numpy() * ny * nz
            H8 = self.c5.cpu().numpy() * (nx * nx - ny * ny)
            H9 = self.c4.cpu().numpy() * nx * ny

            for j in range(0, 3):
                L = fc_light[i, j * 9:(j + 1) * 9]
                top[i, j, :, :] = L[0] * H1 + L[1] * H2 + L[2] * H3 + L[3] * H4 \
                                  + L[4] * H5 + L[5] * H6 + L[6] * H7 + L[7] * H8 + L[8] * H9

        return top

