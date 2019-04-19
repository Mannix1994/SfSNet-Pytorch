# coding=utf8
from __future__ import absolute_import, division, print_function
import os
import cv2
from torch.utils.data import DataLoader
from .base import SfSNetDataset
from config import M
import numpy as np
from os.path import join


class SyntheticDataset(SfSNetDataset):
    def __init__(self, dataset_dir, dataset_ids, size=M):
        super(SyntheticDataset, self).__init__(dataset_dir, size)
        for base_dir in dataset_ids:
            base_id = int(base_dir) - 1
            start_id = base_id * 20 + 1
            for id1 in range(start_id, (base_id + 1) * 20 + 1, 1):
                for id2 in range(1, 4, 1):
                    for id3 in range(1, 6, 1):
                        record = {
                            'albedo': base_dir + '/{:0>6}_albedo_{}_{}.png'.format(id1, id2, id3),
                            'depth': base_dir + '/{:0>6}_depth_{}_{}.png'.format(id1, id2, id3),
                            'face': base_dir + '/{:0>6}_face_{}_{}.png'.format(id1, id2, id3),
                            'light': base_dir + '/{:0>6}_light_{}_{}.txt'.format(id1, id2, id3),
                            'normal': base_dir + '/{:0>6}_normal_{}_{}.png'.format(id1, id2, id3),
                            'mask': base_dir + '/{:0>6}_mask_{}_{}.png'.format(id1, id2, id3),
                            'label': 0,  # label, always be 0
                        }
                        # print(record)
                        self.records.append(record)

    def _transform(self, image_path):
        image = cv2.imread(join(self.dataset_dir, image_path))
        image = cv2.resize(image, dsize=(self.size, self.size))
        image = np.float32(image)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, item):
        record = self.records[item]
        albedo = self._transform(record['albedo'])
        face = self._transform(record['face'])
        normal = self._transform(record['normal'])
        mask = self._transform(record['mask'])
        fc_light = np.array(np.loadtxt(join(self.dataset_dir, record['light'])), dtype=np.float32)

        return self.to_tensor(face=face, mask=mask, normal=normal, albedo=albedo,
                              fc_light=fc_light, label=record['label'])


def prepare_sfsnet_dataset(dataset_dir, size=M):
    ids = sorted(os.listdir(dataset_dir))
    np.random.shuffle(ids)
    # get 10% of ids as test dataset, the rest as train dataset
    train_ids = ids[0:int(0.9 * len(ids))]
    test_ids = ids[int(0.9 * len(ids)):]
    assert len(train_ids) + len(test_ids) == len(ids)
    train_dset = SyntheticDataset(dataset_dir, train_ids, size)
    test_dset = SyntheticDataset(dataset_dir, test_ids, size)
    return train_dset, test_dset


class PreprocessSyntheticDataset(SyntheticDataset):
    def __init__(self, dataset_dir, save_dir, size=M):
        dataset_ids = sorted(os.listdir(dataset_dir))
        super(PreprocessSyntheticDataset, self).__init__(dataset_dir, dataset_ids, size)
        self.__save_dir = save_dir
        for did in dataset_ids:
            if not os.path.exists(join(self.__save_dir, did)):
                os.makedirs(join(self.__save_dir, did))

    def __getitem__(self, item):
        record = self.records[item]
        print(item, record['face'])
        albedo = self._transform(record['albedo'])
        face = self._transform(record['face'])
        normal = self._transform(record['normal'])
        mask = self._transform(record['mask'])
        fc_light = np.array(np.loadtxt(join(self.dataset_dir, record['light'])), dtype=np.float32)

        np.save(join(self.__save_dir, record['albedo'].replace('.png', '.npy')), albedo)
        np.save(join(self.__save_dir, record['face'].replace('.png', '.npy')), face)
        np.save(join(self.__save_dir, record['normal'].replace('.png', '.npy')), normal)
        np.save(join(self.__save_dir, record['mask'].replace('.png', '.npy')), mask)
        np.save(join(self.__save_dir, record['light'].replace('.txt', '.npy')), fc_light)
        return 1


def preprocess_sfsnet_dataset(dataset_dir, save_dir, size=M):
    import multiprocessing

    pd = PreprocessSyntheticDataset(dataset_dir, save_dir, size)
    dl = DataLoader(pd, 64, num_workers=multiprocessing.cpu_count())
    for d in dl:
        pass
