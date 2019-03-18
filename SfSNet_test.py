# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
from torch.nn import init, Parameter
import pickle as pkl

from src.models.model import SfSNet


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
    net.apply_weights_from_pkl('wow/weights.pkl')

    f = open('wow/weights.pkl', 'rb')
    name_weights = pkl.load(f)
    print(name_weights['conv1']['weight'][0, 0, 0, :])
    for name, param in list(net.named_parameters())[0:2]:
        print(name, param.size())
        assert isinstance(name, str)
        if name.endswith('weight'):
            print(param.detach().numpy()[0, 0, 0, :])
