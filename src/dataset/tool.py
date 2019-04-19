# coding=utf8
from __future__ import absolute_import, division, print_function
from typing import Optional
from torch.utils.data import Dataset

from config import M
from .synthetic import SyntheticDataset, prepare_sfsnet_dataset
from .real import CelabaDataset, prepare_celaba_dataset


class ProcessedDataset(Dataset):
    def __init__(self, real, synthetic):
        # type: (Optional[CelabaDataset], Optional[SyntheticDataset]) -> None
        self.__records = []
        if synthetic:
            for record in synthetic.records:
                record['data_dir'] = synthetic.dataset_dir
                record['albedo'] = record['albedo'].replace('.png', '.npy')
                record['face'] = record['face'].replace('.png', '.npy')
                record['normal'] = record['normal'].replace('.png', '.npy')
                record['mask'] = record['mask'].replace('.png', '.npy')
                record['light'] = record['light'].replace('.txt', '.npy')
                self.__records.append(record)
                # print(record)
        if real:
            for record in real.records:
                record['data_dir'] = real.dataset_dir
                self.__records.append(record)
                # print(record)

    @property
    def records(self):
        return self.__records

    def __getitem__(self, item):
        record = self.records[item]
        return CelabaDataset.load(record['data_dir'], record)

    def __len__(self):
        return len(self.__records)


def prepare_processed_dataset(celab_dir, sfs_dir, size=M):
    if celab_dir and sfs_dir:
        s_train, s_test = prepare_sfsnet_dataset(sfs_dir, size)
        r_train, r_test = prepare_celaba_dataset(celab_dir, size)
        train_set = ProcessedDataset(r_train, s_train)
        test_set = ProcessedDataset(r_test, s_test)
    elif celab_dir:
        r_train, r_test = prepare_celaba_dataset(celab_dir, size)
        train_set = ProcessedDataset(r_train, None)
        test_set = ProcessedDataset(r_test, None)
    elif sfs_dir:
        s_train, s_test = prepare_sfsnet_dataset(sfs_dir, size)
        train_set = ProcessedDataset(None, s_train)
        test_set = ProcessedDataset(None, s_test)
    else:
        raise RuntimeError("Both celab_dir and sfs_dir is None???")
    return train_set, test_set
