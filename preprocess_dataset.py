# coding=utf-8
from __future__ import absolute_import, division, print_function
from config import *
from src.dataset.real import preproccess_celaba_dataset
from src.dataset.synthetic import preprocess_sfsnet_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', default=0, type=int,
                        help='define which dataset to preprocess. synthetic or real(CElABA)')
    parser.add_argument('-w', '--weights', help='define the path weights file from stage 0')
    arg = parser.parse_args()

    if arg.stage == 0:
        print(arg)
        preprocess_sfsnet_dataset(SFSNET_DATASET_DIR, SFSNET_DATASET_DIR_NPY, M)
    elif arg.stage == 1 and arg.weights:
        print(arg)
        preproccess_celaba_dataset(CELABA_DATASET_DIR, CELABA_DATASET_DIR_NPY, arg.weights, debug=True)
    else:
        raise RuntimeError("Wrong Option")
