# coding=utf-8
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from torch import from_numpy
from config import M


class SfSNetDataset(Dataset):
    def __init__(self, dataset_dir, size=M):
        assert dataset_dir != '' and dataset_dir is not None
        assert isinstance(size, int) and size > 0
        self.__dataset_dir = dataset_dir
        self.__size = size
        self.__records = []

    @property
    def dataset_dir(self):
        return self.__dataset_dir

    @property
    def size(self):
        return self.__size

    @property
    def records(self):
        return self.__records

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return len(self.__records)

    @staticmethod
    def to_tensor(face, mask, normal, albedo, fc_light, label):
        return from_numpy(face), from_numpy(mask), from_numpy(normal), \
               from_numpy(albedo), from_numpy(fc_light), \
               from_numpy(np.array([label, ], dtype=np.float32))