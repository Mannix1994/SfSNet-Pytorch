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

    def _transform(self, image_path):
        image = cv2.imread(join(self.__dataset_dir, image_path))
        image = cv2.resize(image, dsize=(self.__size, self.__size))
        image = np.float32(image)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, item):
        record = self.__records[item]
        albedo = self._transform(record['albedo'])
        face = self._transform(record['face'])
        normal = self._transform(record['normal'])
        mask = self._transform(record['mask'])
        fc_light = np.array(np.loadtxt(join(self.__dataset_dir, record['light'])), dtype=np.float32)
        return from_numpy(face), from_numpy(mask), from_numpy(normal), \
               from_numpy(albedo), from_numpy(fc_light), \
               from_numpy(np.array([record['label'], ], dtype=np.float32))

    def __len__(self):
        return len(self.__records)