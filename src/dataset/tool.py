# coding=utf8
from __future__ import absolute_import, division, print_function
from typing import Optional
from os.path import join
from torch import from_numpy
import numpy as np
from torch.utils.data import Dataset

from config import M
from .synthetic import SyntheticDataset, prepare_synthetic_dataset
from .real import CelabaDataset, prepare_celaba_dataset


class ProcessedDataset(Dataset):
    def __init__(self, real, synthetic):
        # type: (Optional[CelabaDataset], Optional[SyntheticDataset]) -> None
        self.__records = []
        if synthetic:
            for record in synthetic.records:
                record['albedo'] = join(synthetic.dataset_dir, record['albedo'].replace('.png', '.npy'))
                record['face'] = join(synthetic.dataset_dir, record['face'].replace('.png', '.npy'))
                record['normal'] = join(synthetic.dataset_dir, record['normal'].replace('.png', '.npy'))
                record['mask'] = join(synthetic.dataset_dir, record['mask'].replace('.png', '.npy'))
                record['light'] = join(synthetic.dataset_dir, record['light'].replace('.png', '.npy'))
                self.__records.append(record)
                print(record)
        if real:
            for record in real.records:
                record['albedo'] = join(real.dataset_dir, record['albedo'])
                record['face'] = join(real.dataset_dir, record['face'])
                record['normal'] = join(real.dataset_dir, record['normal'])
                record['mask'] = join(real.dataset_dir, record['mask'])
                record['light'] = join(real.dataset_dir, record['light'])
                self.__records.append(record)
                print(record)

    @property
    def records(self):
        return self.__records

    def __getitem__(self, item):
        record = self.records[item]
        albedo = np.load(record['albedo'])
        face = np.load(record['face'])
        normal = np.load(record['normal'])
        mask = np.load(record['mask'])
        fc_light = np.load(record['mask'])

        return from_numpy(face), from_numpy(mask), from_numpy(normal), \
               from_numpy(albedo), from_numpy(fc_light), \
               from_numpy(np.array([1, ], dtype=np.float32))

    def __len__(self):
        return len(self.__records)


def prepare_processed_dataset(celab_dir, sfs_dir, size=M):
    if sfs_dir and sfs_dir:
        s_train, s_test = prepare_synthetic_dataset(sfs_dir, size)
        r_train, r_test = prepare_celaba_dataset(celab_dir, size)
        train_set = ProcessedDataset(r_train, s_train)
        test_set = ProcessedDataset(r_test, s_test)
    elif celab_dir:
        r_train, r_test = prepare_celaba_dataset(celab_dir, size)
        train_set = ProcessedDataset(r_train, None)
        test_set = ProcessedDataset(r_test, None)
    elif sfs_dir:
        s_train, s_test = prepare_synthetic_dataset(sfs_dir, size)
        train_set = ProcessedDataset(None, s_train)
        test_set = ProcessedDataset(None, s_test)
    else:
        raise RuntimeError("Both celab_dir and sfs_dir is None???")
    return train_set, test_set
