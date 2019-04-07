# coding=utf-8
from __future__ import absolute_import, division, print_function

from src import process_dataset
from config import SFSNET_DATASET_DIR, SFSNET_DATASET_DIR_NPY

if __name__ == '__main__':
    process_dataset(SFSNET_DATASET_DIR, SFSNET_DATASET_DIR_NPY, 128)
