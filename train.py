# coding=utf-8
from __future__ import absolute_import, division, print_function

import torch
from src import *
from torch.utils.data import DataLoader
from config import SFSNET_DATASET_DIR, SFSNET_DATASET_DIR_NPY
import os
import time
from config import PROJECT_DIR

if __name__ == '__main__':
    pass
    # model = SfSNet()
    # with open('data/temp_1554633573.72.pth', 'r') as f:
    #     model.load_state_dict(torch.load(f))
    # exit()


def weight_init(layer):
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        torch.nn.init.constant_(layer.bias.data, 0.)
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(layer.weight.data)


def train():
    data_dir = os.path.join(PROJECT_DIR, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    t = time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime(time.time()))
    sta = Statistic('data/train_%s.csv' % t, True, 'epoch', 'step', 'total_step', 'learning_rate', 'loss')

    # define batch size
    batch_size = 32
    # define net
    model = SfSNet()
    # init weights
    model.apply(weight_init)
    # load last trained weight
    with open('data/temp_2019.04.09_19.51.44.pth', 'r') as f:
        model.load_state_dict(torch.load(f))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [62, ...] -> [32, ...], [32, ...] on 2 GPUs
        model = torch.nn.DataParallel(model).cuda()
        # set batch size to 64
        batch_size = 64
    if torch.cuda.is_available():
        model = model.cuda()

    # set to train mode
    model.train()

    # define dataset
    # train_dset, test_dset = prepare_dataset(SFSNET_DATASET_DIR)
    train_dset, test_dset = prepare_processed_dataset(SFSNET_DATASET_DIR_NPY, size=128)

    # define dataloader
    dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    # learning rate scheduler
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000, 6000, 10000], gamma=0.1)
    # lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, verbose=True)
    # lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, 1e-4)

    l2_layer = L2LossLayerWt(0.1, 0.1)
    l1_layer = L1LossLayerWt(0.5, 0.5)
    normal_layer = NormLayer()
    change_form_layer = ChangeFormLayer()
    if torch.cuda.is_available():
        shading_layer = ShadingLayer(gpu=True)
    else:
        shading_layer = ShadingLayer(gpu=False)

    image_size = 128*128*3.
    step_size = int(len(train_dset)/batch_size)
    try:
        for epoch in range(500):
            # fc_light_gt = label
            # label3 = label1 = label2
            print('*' * 100)
            print("epoch: ", epoch)
            for step, (data, mask, normal, albedo, fc_light_gt, label) in enumerate(dloader):
                lr_sch.step(epoch * step_size + step)
                if torch.cuda.is_available():
                    data = data.cuda()
                    mask = mask.cuda()
                    normal = normal.cuda()
                    albedo = albedo.cuda()
                    fc_light_gt = fc_light_gt.cuda()
                    label = label.cuda()
                # forward net
                Nconv0, Acov0, fc_light = model(data)
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
                mask_recon = recon * maskch

                datach = change_form_layer(data)
                mask_data = datach * maskch

                reconloss = l1_layer(mask_recon, mask_data, label)
                # -------------aloss----------
                arec = Acov0 * mask
                albedo_m = albedo * mask
                aloss = l1_layer(arec, albedo_m, label)
                # -----------loss--------------
                nrec = Nconv0 * mask
                normal_m = normal * mask
                loss = l1_layer(nrec, normal_m, label)
                # ------------
                lloss = l2_layer(fc_light, fc_light_gt, label)

                total_loss = reconloss/image_size + aloss/image_size + loss/image_size + lloss/27
                # backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # save train log
                record = [epoch, step, epoch * step_size + step, optimizer.param_groups[0]['lr'],
                          total_loss.cpu().detach().numpy()]
                sta.add(*record)
                print(*record)

    except KeyboardInterrupt as e:
        print("用户主动退出...")
        pass
    except:
        print("其它异常...")
        raise
    finally:
        sta.save()
        t = time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime(time.time()))
        with open('data/temp_%s.pth' % t, 'w') as f:
            torch.save(model.module.state_dict(), f)


if __name__ == '__main__':
    train()
