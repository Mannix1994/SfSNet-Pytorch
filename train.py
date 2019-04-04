# coding=utf-8
from __future__ import absolute_import, division, print_function

import torch
import cv2
from src import *
from torch.utils.data import DataLoader
from config import SFSNET_DATASET_DIR
import numpy as np

if __name__ == '__main__':
    pass


def train():
    # define net
    net = SfSNet().cuda()
    # set to train mode
    net.train()

    # define dataset
    train_dset, test_dset = prepare_dataset(SFSNET_DATASET_DIR)

    # define dataloader
    dloader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=16)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    l2_layer = L2LossLayerWt(0.1, 0.1)
    l1_layer = L1LossLayerWt(0.5, 0.5)
    normal_layer = NormLayer()
    change_form_layer = ChangeFormLayer()
    shading_layer = ShadingLayer(gpu=net.is_cuda)

    try:
        for epoch in range(1000):
            # fc_light_gt = label
            # label3 = label1 = label2
            for step, (data, mask, normal, albedo, fc_light_gt, label) in enumerate(dloader):
                data = data.cuda()
                mask = mask.cuda()
                normal = normal.cuda()
                albedo = albedo.cuda()
                fc_light_gt = fc_light_gt.cuda()
                label = label.cuda()
                # forward net
                Nconv0, Acov0, fc_light = net(data)
                # ---------compute reconloss------------
                # normalize
                normalize = normal_layer(Nconv0)
                # change channel of normal
                norch1 = change_form_layer(normalize)
                # compute shading
                shading = shading_layer(norch1, fc_light)
                # change channel od albedo
                albech2 = change_form_layer(Acov0)
                # get recon images
                recon = albech2 * shading
                # change channel format
                maskch4 = change_form_layer(mask)
                # compute mask with recon
                mask_recon = recon * mask

                datach3 = change_form_layer(data)
                mask_data = datach3 * maskch4

                reconloss = l1_layer(mask_recon, mask_data, label)
                # -------------aloss----------
                mask_al = Acov0 * mask
                mask_algt = albedo * mask
                aloss = l1_layer(mask_al, mask_algt, label)
                # -----------loss--------------
                mask_nor = Nconv0 * mask
                mask_norgt = normal * mask
                loss = l1_layer(mask_nor, mask_norgt, label)
                # ------------
                lloss = l2_layer(fc_light, fc_light_gt, label)

                total_loss = reconloss + aloss + loss + lloss
                # backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                print(total_loss)

    except KeyboardInterrupt as e:
        print("用户主动退出...")
        pass
    finally:
        with open('data/temp.pth', 'w') as f:
            torch.save(net.state_dict(), f)


if __name__ == '__main__':
    train()
