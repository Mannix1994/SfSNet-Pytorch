# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import cv2
import torch
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from torch import from_numpy


class SfSNetDataset(Dataset):
    def __init__(self, dataset_dir, dataset_ids, size=128):
        assert dataset_dir != '' and dataset_dir is not None
        assert isinstance(size, int) and size > 0
        self.__dataset_dir = dataset_dir
        self.__size = size
        self.__records = []
        for base_dir in dataset_ids:
            base_id = int(base_dir) - 1
            start_id = base_id * 20 + 1
            for id1 in range(start_id, (base_id + 1) * 20 + 1, 1):
                for id2 in range(1, 4, 1):
                    for id3 in range(1, 6, 1):
                        record = (
                            base_dir + '/{:0>6}_albedo_{}_{}.png'.format(id1, id2, id3),
                            base_dir + '/{:0>6}_depth_{}_{}.png'.format(id1, id2, id3),
                            base_dir + '/{:0>6}_face_{}_{}.png'.format(id1, id2, id3),
                            base_dir + '/{:0>6}_light_{}_{}.txt'.format(id1, id2, id3),
                            base_dir + '/{:0>6}_normal_{}_{}.png'.format(id1, id2, id3),
                            base_dir + '/{:0>6}_mask_{}_{}.png'.format(id1, id2, id3),
                            1,  # label, always be 1
                        )
                        # print(record)
                        self.__records.append(record)

    def __transform(self, image_path):
        image = cv2.imread(join(self.__dataset_dir, image_path))
        image = cv2.resize(image, dsize=(self.__size, self.__size))
        image = np.float32(image)
        image /= 255.0
        image = np.transpose(image, (2, 1, 0))
        return image

    def __getitem__(self, item):
        record = self.__records[item]
        albedo = self.__transform(record[0])
        face = self.__transform(record[2])
        normal = self.__transform(record[4])
        mask = self.__transform(record[5])
        fc_light = np.array(np.loadtxt(join(self.__dataset_dir, record[3])), dtype=np.float32)
        return from_numpy(face), from_numpy(mask), from_numpy(normal), \
               from_numpy(albedo), from_numpy(fc_light), \
               from_numpy(np.array([record[6], ], dtype=np.float32))

    def __len__(self):
        return len(self.__records)


def prepare_dataset(dataset_dir, size=128):
    ids = sorted(os.listdir(dataset_dir))
    np.random.shuffle(ids)
    # get 10% of ids as test dataset, the rest as train dataset
    train_ids = ids[0:int(0.9 * len(ids))]
    test_ids = ids[int(0.9 * len(ids)):]
    assert len(train_ids) + len(test_ids) == len(ids)
    train_dset = SfSNetDataset(dataset_dir, train_ids, size)
    test_dset = SfSNetDataset(dataset_dir, test_ids, size)
    return train_dset, test_dset


if __name__ == '__main__':
    train_dset, test_dset = prepare_dataset('/home/creator/Data/DATA_pose_15')
    print(len(train_dset), len(test_dset))
