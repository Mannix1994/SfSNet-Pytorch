# coding=utf-8
import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

# image's size, DO NOT CHANGE!
M = 128  # size of input for SfSNet

# landmarks's path
LANDMARK_PATH = os.path.join(PROJECT_DIR, 'data/shape_predictor_68_face_landmarks.dat')
