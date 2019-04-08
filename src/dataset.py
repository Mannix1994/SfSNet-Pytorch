# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import cv2
import torch
import numpy as np
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy


class SfSNetDataset(Dataset):
    def __init__(self, dataset_dir, dataset_ids, size=128):
        assert dataset_dir != '' and dataset_dir is not None
        assert isinstance(size, int) and size > 0
        self._dataset_dir = dataset_dir
        self._size = size
        self._records = []
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
                            'label': 1,  # label, always be 1
                        }
                        # print(record)
                        self._records.append(record)

    def _transform(self, image_path):
        image = cv2.imread(join(self._dataset_dir, image_path))
        image = cv2.resize(image, dsize=(self._size, self._size))
        image = np.float32(image)
        image /= 255.0
        image = np.transpose(image, (2, 1, 0))
        return image

    def __getitem__(self, item):
        record = self._records[item]
        albedo = self._transform(record['albedo'])
        face = self._transform(record['face'])
        normal = self._transform(record['normal'])
        mask = self._transform(record['mask'])
        fc_light = np.array(np.loadtxt(join(self._dataset_dir, record['light'])), dtype=np.float32)
        return from_numpy(face), from_numpy(mask), from_numpy(normal), \
               from_numpy(albedo), from_numpy(fc_light), \
               from_numpy(np.array([record['label'], ], dtype=np.float32))

    def __len__(self):
        return len(self._records)


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


class PreprocessDataset(SfSNetDataset):
    def __init__(self, dataset_dir, save_dir, size=128):
        dataset_ids = sorted(os.listdir(dataset_dir))
        super(PreprocessDataset, self).__init__(dataset_dir, dataset_ids, size)
        self._save_dir = save_dir
        for did in dataset_ids:
            if not os.path.exists(join(self._save_dir, did)):
                os.makedirs(join(self._save_dir, did))

    def __getitem__(self, item):
        record = self._records[item]
        print(item, record['face'])
        albedo = self._transform(record['albedo'])
        face = self._transform(record['face'])
        normal = self._transform(record['normal'])
        mask = self._transform(record['mask'])
        fc_light = np.array(np.loadtxt(join(self._dataset_dir, record['light'])), dtype=np.float32)

        np.save(join(self._save_dir, record['albedo'].replace('.png', '.npy')), albedo)
        np.save(join(self._save_dir, record['face'].replace('.png', '.npy')), face)
        np.save(join(self._save_dir, record['normal'].replace('.png', '.npy')), normal)
        np.save(join(self._save_dir, record['mask'].replace('.png', '.npy')), mask)
        np.save(join(self._save_dir, record['light'].replace('.txt', '.npy')), fc_light)
        return fc_light


def process_dataset(dataset_dir, save_dir, size=128):
    import multiprocessing

    pd = PreprocessDataset(dataset_dir, save_dir, size)
    dl = DataLoader(pd, 16, num_workers=multiprocessing.cpu_count())
    for d in dl:
        pass


class ProcessedDataset(SfSNetDataset):
    def __init__(self, save_dir, dataset_ids, size=128):
        super(ProcessedDataset, self).__init__(save_dir, dataset_ids, size)
        self._save_dir = save_dir

    def __getitem__(self, item):
        record = self._records[item]
        albedo = np.load(join(self._save_dir, record['albedo'].replace('.png', '.npy')))
        face = np.load(join(self._save_dir, record['face'].replace('.png', '.npy')))
        normal = np.load(join(self._save_dir, record['normal'].replace('.png', '.npy')))
        mask = np.load(join(self._save_dir, record['mask'].replace('.png', '.npy')))
        fc_light = np.load(join(self._save_dir, record['light'].replace('.txt', '.npy')))

        return from_numpy(face), from_numpy(mask), from_numpy(normal), \
               from_numpy(albedo), from_numpy(fc_light), \
               from_numpy(np.array([1, ], dtype=np.float32))


def prepare_processed_dataset(save_dir, size=128):
    ids = sorted(os.listdir(save_dir))
    np.random.shuffle(ids)
    # get 10% of ids as test dataset, the rest as train dataset
    train_ids = ids[0:int(0.9 * len(ids))]
    test_ids = ids[int(0.9 * len(ids)):]
    assert len(train_ids) + len(test_ids) == len(ids)
    train_dset = ProcessedDataset(save_dir, train_ids, size)
    test_dset = ProcessedDataset(save_dir, test_ids, size)
    return train_dset, test_dset


if __name__ == '__main__':
    from config import SFSNET_DATASET_DIR, SFSNET_DATASET_DIR_NPY
    train_dset, test_dset = prepare_dataset(SFSNET_DATASET_DIR)
    print(len(train_dset), len(test_dset))

    train_dset, test_dset = prepare_processed_dataset(SFSNET_DATASET_DIR_NPY)
    print(len(train_dset), len(test_dset))


