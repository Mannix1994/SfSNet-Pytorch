# coding=utf-8
from __future__ import absolute_import, division, print_function
import caffe
import numpy as np
import cv2


def read_image(path):
    # type: (str) -> np.ndarray
    # 读取图像
    image = cv2.imread(path)
    # 调整大小
    image = cv2.resize(image, (128, 128))
    # 缩放到0~1
    image = np.float32(image)/255.0
    # (128, 128, 3) to (3, 128, 128)
    image = np.transpose(image, [2, 0, 1])
    # (128, 128, 3) to (1, 3, 128, 128)
    image = np.expand_dims(image, 0)

    return image


if __name__ == '__main__':

    # prototxt文件
    MODEL_FILE = 'SfSNet-Caffe/SfSNet_deploy.prototxt'
    # 预先训练好的caffe模型
    PRETRAIN_FILE = 'SfSNet-Caffe/SfSNet.caffemodel.h5'
    # 定义网络
    net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
    # 读取并预处理图像
    im = read_image('data/1.png_face.png')
    # 前向传播
    out = net.forward(end='conv3')

    print(out.keys())
    # 保存
    np.save('conv3.caffe.npy', out['conv3'])
