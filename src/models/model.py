# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import torchvision
import pickle as pkl
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        # nbn1/nbn2/.../nbn5 abn1/abn2/.../abn5
        self.bn = nn.BatchNorm2d(in_channel)
        # nconv1/nconv2/.../nconv5 aconv1/aconv2/.../aconv5
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        # nbn1r/nbn2r/.../nbn5r abn1r/abn2r/.../abn5r
        self.bnr = nn.BatchNorm2d(out_channel)
        # nconv1r/nconv2r/.../nconv5r aconv1r/aconv2r/.../anconv5r
        self.convr = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.convr(F.relu(self.bnr(out)))
        out += x
        return out


class SfSNet(nn.Module):  # SfSNet = PS-Net in SfSNet_deploy.prototxt
    def __init__(self):
        # C64
        super(SfSNet, self).__init__()
        # TODO 初始化器 xavier
        self.conv1 = nn.Conv2d(3, 64, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(64)
        # C128
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        # C128 S2
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        # ------------RESNET for normals------------
        # RES1
        self.n_res1 = ResidualBlock(128, 128)
        # RES2
        self.n_res2 = ResidualBlock(128, 128)
        # RES3
        self.n_res3 = ResidualBlock(128, 128)
        # RES4
        self.n_res4 = ResidualBlock(128, 128)
        # RES5
        self.n_res5 = ResidualBlock(128, 128)
        # nbn6r
        self.nbn6r = nn.BatchNorm2d(128)
        # CD128
        # TODO 初始化器 bilinear
        self.nup6 = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128)
        # nconv6
        self.nconv6 = nn.Conv2d(128, 128, 1, 1, 0)
        # nbn6
        self.nbn6 = nn.BatchNorm2d(128)
        # CD 64
        self.nconv7 = nn.Conv2d(128, 64, 3, 1, 1)
        # nbn7
        self.nbn7 = nn.BatchNorm2d(64)
        # C*3
        self.nconv0 = nn.Conv2d(64, 3, 1, 1, 0)

        # --------------------Albedo---------------
        # RES1
        self.a_res1 = ResidualBlock(128, 128)
        # RES2
        self.a_res2 = ResidualBlock(128, 128)
        # RES3
        self.a_res3 = ResidualBlock(128, 128)
        # RES4
        self.a_res4 = ResidualBlock(128, 128)
        # RES5
        self.a_res5 = ResidualBlock(128, 128)
        # abn6r
        self.abn6r = nn.BatchNorm2d(128)
        # CD128
        self.aup6 = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128)
        # nconv6
        self.aconv6 = nn.Conv2d(128, 128, 1, 1, 0)
        # nbn6
        self.abn6 = nn.BatchNorm2d(128)
        # CD 64
        self.aconv7 = nn.Conv2d(128, 64, 3, 1, 1)
        # nbn7
        self.abn7 = nn.BatchNorm2d(64)
        # C*3
        self.aconv0 = nn.Conv2d(64, 3, 1, 1, 0)

        # ---------------Light------------------
        # lconv1
        self.lconv1 = nn.Conv2d(384, 128, 1, 1, 0)
        # lbn1
        self.lbn1 = nn.BatchNorm2d(128)
        # lpool2r
        self.lpool2r = nn.AvgPool2d(64)
        # fc_light
        self.fc_light = nn.Linear(128, 27)

    def forward(self, inputs):
        # C64
        x = F.relu(self.bn1(self.conv1(inputs)))
        # C128
        x = F.relu(self.bn2(self.conv2(x)))
        # C128 S2
        conv3 = self.conv3(x)
        # ------------RESNET for normals------------
        # RES1
        x = self.n_res1(conv3)
        # RES2
        x = self.n_res2(x)
        # RES3
        x = self.n_res3(x)
        # RES4
        x = self.n_res4(x)
        # RES5
        nsum5 = self.n_res5(x)
        # nbn6r
        x = F.relu(self.nbn6r(nsum5))
        # CD128
        x = self.nup6(x)
        # nconv6/nbn6/nrelu6
        x = F.relu(self.nbn6(self.nconv6(x)))
        # nconv7/nbn7/nrelu7
        x = F.relu(self.nbn7(self.nconv7(x)))
        # nconv0
        normal = self.nconv0(x)
        # --------------------Albedo---------------
        # RES1
        x = self.a_res1(conv3)
        # RES2
        x = self.a_res2(x)
        # RES3
        x = self.a_res3(x)
        # RES4
        x = self.a_res4(x)
        # RES5
        asum5 = self.a_res5(x)
        # nbn6r
        x = F.relu(self.abn6r(asum5))
        # CD128
        x = self.aup6(x)
        # nconv6/nbn6/nrelu6
        x = F.relu(self.abn6(self.aconv6(x)))
        # nconv7/nbn7/nrelu7
        x = F.relu(self.abn7(self.aconv7(x)))
        # nconv0
        albedo = self.aconv0(x)

        # ---------------Light------------------
        # lconcat1, shape(1 256 64 64)
        x = torch.cat([nsum5, asum5], 1)
        # lconcat2, shape(1 384 64 64)
        x = torch.cat([x, conv3], 1)
        # lconv1/lbn1/lrelu1 shape(1 128 64 64)
        x = F.relu(self.lbn1(self.lconv1(x)))
        # lpool2r, shape(1 128 1 1)
        x = self.lpool2r(x)
        x = x.view(-1, 128)
        # fc_light
        light = self.fc_light(x)

        return normal, albedo, light

    def load_weights_from_pkl(self, weights_pkl):
        from torch import from_numpy
        with open(weights_pkl, 'rb') as wp:
            name_weights = pkl.load(wp)

            def _set_deconv(layer, key):
                layer.weight.data = from_numpy(name_weights[key]['weight'])

            def _set_bn(layer, key):
                layer.weight.data = from_numpy(name_weights[key]['bias'])
                layer.bias.data = from_numpy(name_weights[key]['weight'])

            def _set(layer, key):
                layer.weight.data = from_numpy(name_weights[key]['weight'])
                layer.bias.data = from_numpy(name_weights[key]['bias'])

            def _set_res(layer, n_or_a, index):
                _set(layer.bn, n_or_a + 'bn' + str(index))
                _set(layer.conv, n_or_a + 'conv' + str(index))
                _set(layer.bnr, n_or_a + 'bn' + str(index) + 'r')
                _set(layer.convr, n_or_a + 'conv' + str(index) + 'r')
            _set(self.conv1, 'conv1')
            _set(self.bn1, 'bn1')
            _set(self.conv2, 'conv2')
            _set(self.bn2, 'bn2')
            _set(self.conv3, 'conv3')
            _set_res(self.n_res1, 'n', 1)
            _set_res(self.n_res2, 'n', 2)
            _set_res(self.n_res3, 'n', 3)
            _set_res(self.n_res4, 'n', 4)
            _set_res(self.n_res5, 'n', 5)
            _set(self.nbn6r, 'nbn6r')
            _set_deconv(self.nup6, 'nup6')
            _set(self.nconv6, 'nconv6')
            _set(self.nbn6, 'nbn6')
            _set(self.nconv7, 'nconv7')
            _set(self.nbn7, 'nbn7')
            _set(self.nconv0, 'Nconv0')
            _set_res(self.a_res1, 'a', 1)
            _set_res(self.a_res2, 'a', 2)
            _set_res(self.a_res3, 'a', 3)
            _set_res(self.a_res4, 'a', 4)
            _set_res(self.a_res5, 'a', 5)
            _set(self.abn6r, 'abn6r')
            _set_deconv(self.aup6, 'aup6')
            _set(self.aconv6, 'aconv6')
            _set(self.abn6, 'abn6')
            _set(self.aconv7, 'aconv7')
            _set(self.abn7, 'abn7')
            _set(self.aconv0, 'Aconv0')
            _set(self.lconv1, 'lconv1')
            _set(self.lbn1, 'lbn1')
            _set(self.fc_light, 'fc_light')


if __name__ == '__main__':
    net = SfSNet()
    net.eval()

    print(len(list(net.named_parameters())))
    for name, param in list(net.named_parameters()):
        print(name, param.size())
