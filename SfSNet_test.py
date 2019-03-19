# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import cv2
import numpy as np
import pickle as pkl
from torch.nn import init, Parameter
from src.models.model import SfSNet
from config import M


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    net = SfSNet()
    net.eval()
    net.load_weights_from_pkl('wow/weights.pkl')

    image = cv2.imread('1.png_face.png')
    im = cv2.resize(image, (M, M))
    im = np.float32(im) / 255.0
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, [2, 0, 1])  # from (128, 128, 3) to (1, 3, 128, 128)
    im = np.expand_dims(im, 0)
    print(np.min(im), np.max(im))

    x = net(torch.from_numpy(im))

    print(x[2].detach().numpy())

    # print(net.bn1.state_dict().keys())
    # for name, parm in net.bn1.state_dict().items():
    #     print(name, parm.size())
    # for name, parm in net.named_parameters():
    #     print(name)

