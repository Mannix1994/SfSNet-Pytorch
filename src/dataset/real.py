# coding=utf8
from __future__ import absolute_import, division, print_function
import os
from .base import SfSNetDataset
from config import M, LANDMARK_PATH
import cv2
import torch
from src import *
from ..tools import SfSNetEval
from os.path import join
import numpy as np


class CelabaDataset(SfSNetDataset):
    def __init__(self, dataset_dir, image_list, size=M):
        super(CelabaDataset, self).__init__(dataset_dir, size)
        for image in image_list:
            record = {
                'albedo': image+'_albedo.npy',
                'depth': image+'_depth.npy',
                'face': image+'_face.npy',
                'light': image+'_light.npy',
                'normal': image+'_normal.npy',
                'mask': image+'_mask.npy',
                'label': 1,  # label, always be 1
            }
            # print(record)
            self.records.append(record)
            # print(record)

    def __getitem__(self, item):
        record = self.records[item]
        return self.load(self.dataset_dir, record)

    @staticmethod
    def load(data_dir, record):
        albedo = np.load(join(data_dir, record['albedo']))
        face = np.load(join(data_dir, record['face']))
        normal = np.load(join(data_dir, record['normal']))
        mask = np.load(join(data_dir, record['mask']))
        fc_light = np.load(join(data_dir, record['light']))

        return SfSNetDataset.to_tensor(face=face, mask=mask, normal=normal,
                                       albedo=albedo, fc_light=fc_light, label=record['label'])


def prepare_celaba_dataset(dataset_dir, size=M):
    with open(os.path.join(dataset_dir, 'list.txt'), 'r') as f:
        images = [line.strip() for line in f]
        # get 10% of ids as test dataset, the rest as train dataset
        train_ids = images[0:int(0.9 * len(images))]
        test_ids = images[int(0.9 * len(images)):]
        assert len(train_ids) + len(test_ids) == len(images)
        train_dset = CelabaDataset(dataset_dir, train_ids, size)
        test_dset = CelabaDataset(dataset_dir, test_ids, size)
        return train_dset, test_dset


def preproccess_celaba_dataset(dataset_dir, save_dir, size=M, debug=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get image list
    image_list = sorted(os.listdir(dataset_dir))
    image_list = [i for i in image_list if i.endswith('.jpg')]

    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    net.load_state_dict(torch.load('data/SfSNet.pth'))
    # define sfsnet tool
    ss = SfSNetEval(net, LANDMARK_PATH)

    f = open(join(save_dir, 'list.txt'), 'w')

    for image_name in image_list:
        # read image
        image = cv2.imread(join(dataset_dir, image_name))
        print(join(dataset_dir, image_name))
        # crop face and generate mask of face
        o_im, face, Irec, n_out2, al_out2, Ishd, mask, fc_light, aligned = ss.predict(image, True)

        if aligned:
            f.write(image_name+'\n')
            np.save(join(save_dir, image_name+'_face.npy'), np.transpose(face, [2, 0, 1]))
            np.save(join(save_dir, image_name+'_normal.npy'), np.transpose(n_out2, [2, 0, 1]))
            np.save(join(save_dir, image_name+'_albedo.npy'), np.transpose(al_out2, [2, 0, 1]))
            np.save(join(save_dir, image_name+'_mask.npy'), np.transpose(mask, [2, 0, 1]).astype(np.float32))
            np.save(join(save_dir, image_name+'_light.npy'), np.squeeze(fc_light, 0))

        if debug:
            cv2.imshow("image", face)
            cv2.imshow("Normal", n_out2)
            cv2.imshow("Albedo", al_out2)
            cv2.imshow("Recon", Irec)
            cv2.imshow("Shading", Ishd)

            cv2.imwrite('shading.png', convert(Irec))
            if cv2.waitKey(1) == 27:
                f.close()
                exit()
    f.close()
