# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import time
import multiprocessing
import pickle
import torch
from src import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from config import SFSNET_DATASET_DIR_NPY
from config import PROJECT_DIR, M
import caffe
import sys
import numpy as np

join = os.path.join
sys.path.append(join(PROJECT_DIR, 'SfSNet-Caffe/SfSNet_train/python'))


def to(*tensors):
    ret = [x.cpu().detach().numpy() for x in tensors]
    return ret


def load_torch_net(mode=True):
    net = SfSNet()
    # load last trained weight
    with open(join(PROJECT_DIR, 'data/SfSNet.pth'), 'rb') as f:
        net.load_state_dict(torch.load(f))

    if mode:
        return net.train()
    else:
        return net.eval()


def load_caffe_net(mode=True):
    # set gpu mode, if you don't have gpu, use caffe.set_mode_cpu()
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    caffe.set_mode_cpu()

    # prototxt文件
    model_file = join(PROJECT_DIR, 'SfSNet-Caffe/SfSNet_deploy_train_compare.prototxt')
    # 预先训练好的caffe模型
    weights = join(PROJECT_DIR, 'SfSNet-Caffe/SfSNet.caffemodel.h5')
    # 定义网络
    if mode:
        net = caffe.Net(model_file, weights, caffe.TRAIN)
    else:
        net = caffe.Net(model_file, weights, caffe.TEST)

    return net


def diff(a, b):
    if not isinstance(a, np.ndarray) and isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    if not isinstance(b, np.ndarray) and isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()
    print(np.sum(np.abs(a - b)))


def train():
    data_dir = os.path.join(PROJECT_DIR, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # define batch size
    batch_size = 32
    # define net
    torch_net = load_torch_net(True)
    caffe_net = load_caffe_net(True)

    # define dataset
    # train_dset, test_dset = prepare_dataset(SFSNET_DATASET_DIR)
    train_dset, test_dset = prepare_processed_dataset(SFSNET_DATASET_DIR_NPY, size=M)

    # define dataloader
    dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())

    l2_layer = L2LossLayerWt(0.1, 0.1)
    l1_layer = L1LossLayerWt(0.5, 0.5)
    normal_layer = NormLayer()
    change_form_layer = ChangeFormLayer()
    shading_layer = ShadingLayer(gpu=False)
    try:
        for epoch in range(0, 500, 1):
            # fc_light_gt = label
            # label3 = label1 = label2
            print('*' * 80)
            print("epoch: ", epoch)
            for step, (data, mask, normal, albedo, fc_light_gt, label) in enumerate(dloader):
                # forward net
                Nconv0, Acov0, fc_light = torch_net(data)
                # ---------compute reconloss------------
                # normalize
                recnormal = normal_layer(Nconv0)
                # change channel of normal
                recnormalch = change_form_layer(recnormal)
                # compute shading
                shading = shading_layer(recnormalch, fc_light)
                # change channel od albedo
                albedoch = change_form_layer(Acov0)
                # get recon images
                recon = albedoch * shading
                # change channel format
                maskch = change_form_layer(mask)
                # compute mask with recon
                recon_mask = recon * maskch

                datach = change_form_layer(data)
                data_mask = datach * maskch

                reconloss = l1_layer(recon_mask, data_mask, label)
                # -------------aloss----------
                arec = Acov0 * mask
                albedo_m = albedo * mask
                aloss = l1_layer(arec, albedo_m, label)
                # -----------loss--------------
                n_rec = Nconv0 * mask
                normal_m = normal * mask
                loss = l1_layer(n_rec, normal_m, label)
                # ------------
                lloss = l2_layer(fc_light, fc_light_gt, label)

                caffe_net.blobs['data'].data[...] = data.numpy()
                caffe_net.blobs['mask'].data[...] = mask.numpy()
                caffe_net.blobs['normal'].data[...] = normal.numpy()
                caffe_net.blobs['albedo'].data[...] = albedo.numpy()
                caffe_net.blobs['label'].data[...] = fc_light_gt.numpy()
                caffe_net.blobs['label2'].data[...] = label.numpy()
                c_losses = caffe_net.forward()

                diff(caffe_net.blobs['Aconv0'].data, Acov0)
                diff(caffe_net.blobs['Nconv0'].data, Nconv0)
                diff(caffe_net.blobs['fc_light'].data, fc_light)
                diff(caffe_net.blobs['recnormal'].data, recnormal)
                diff(caffe_net.blobs['recnormalch'].data, recnormalch)
                diff(caffe_net.blobs['shading'].data, shading)
                diff(caffe_net.blobs['albedoch'].data, albedoch)
                diff(caffe_net.blobs['recon'].data, recon)
                diff(caffe_net.blobs['recon_mask'].data, recon_mask)
                diff(caffe_net.blobs['data_mask'].data, data_mask)
                diff(caffe_net.blobs['label2'].data, label)
                diff(caffe_net.blobs['reconloss'].data, reconloss)
                diff(caffe_net.blobs['reconloss'].data, reconloss)
                diff(caffe_net.blobs['lloss'].data, lloss)
                diff(caffe_net.blobs['loss'].data, loss)
                diff(caffe_net.blobs['aloss'].data, aloss)

                exit()

    except KeyboardInterrupt:
        print("用户主动退出...")
        pass
    except IOError:
        print("其它异常...")
        raise


if __name__ == '__main__':
    train()
