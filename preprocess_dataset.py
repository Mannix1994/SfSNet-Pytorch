# coding=utf-8
from __future__ import absolute_import, division, print_function
from config import *
from src.dataset.real import preproccess_celaba_dataset
from src.dataset.synthetic import preprocess_sfsnet_dataset


if __name__ == '__main__':
    preproccess_celaba_dataset(CELABA_DATASET_DIR, CELABA_DATASET_DIR_NPY, M)
    # preprocess_sfsnet_dataset(SFSNET_DATASET_DIR, SFSNET_DATASET_DIR_NPY, M)

