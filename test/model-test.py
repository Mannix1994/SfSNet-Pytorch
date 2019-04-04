# coding=utf-8
from __future__ import absolute_import, division, print_function
import numpy as np
from src.model import SfSNet
import torch
import cv2

def same(arr1, arr2):
    # type: (np.ndarray, np.ndarray) -> bool
    # 判断shape是否相同
    assert arr1.shape == arr2.shape
    # 对应元素相减求绝对值
    diff = np.abs(arr1 - arr2)
    # 判断是否有任意一个两元素之差小于阈值1e-5
    return (diff < 1e-5).any()


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
    caffe_result = np.load('conv3.caffe.npy')
    torch_result = np.load('conv3.pytorch.npy')
    print(same(caffe_result, torch_result))


if __name__ == '__main__':
    # 新建网络实例
    net = SfSNet()
    # 载入参数
    net.load_weights_from_pkl('SfSNet-Caffe/weights.pkl')
    # 设置为测试模式
    net.eval()
    # 读取并预处理图像
    image = read_image('data/1.png_face.png')
    # 前向传播
    out = net(torch.from_numpy(image))
    # 保存
    np.save('conv3.pytorch.npy', out[0].detach().numpy())