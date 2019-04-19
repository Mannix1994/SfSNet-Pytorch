# coding=utf-8
import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

# image's size, DO NOT CHANGE!
M = 128  # size of input for SfSNet

# landmarks's path
LANDMARK_PATH = os.path.join(PROJECT_DIR, 'data/shape_predictor_68_face_landmarks.dat')

# synthetic SfSNet dataset directory
SFSNET_DATASET_DIR = '/home/creator/Data/DATA_pose_15'

# processed synthetic dataset directory
SFSNET_DATASET_DIR_NPY = '/home/creator/F/DATA_pose_15_npy'

# CelabA dataset directory
CELABA_DATASET_DIR = '/home/creator/Data/img_align_celeba'

# processed CelabA dataset directory
CELABA_DATASET_DIR_NPY = '/home/creator/Data/img_align_celeba_npy'
