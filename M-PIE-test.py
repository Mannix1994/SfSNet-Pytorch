# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import numpy as np
import cv2
import torch
from config import M, LANDMARK_PATH, PROJECT_DIR
from src.functions import create_shading_recon
from src.model import SfSNet
from src.utils import convert

M_PIE_DIR = '/home/creator/Projects/DL/MVCNN-keras-Face-Yale/data/M-PIE/train/001/*.png'


class MaskGenerator(object):
    def __init__(self):
        super(MaskGenerator, self).__init__()

    def align(self,  image, crop_size=(128, 128), scale=3.5):
        image = cv2.copyMakeBorder(image, 70, 70, 70, 70, cv2.BORDER_CONSTANT)
        image = cv2.resize(image, crop_size)
        mask = np.ones(image.shape, dtype=np.uint8)*255
        return mask, image


if __name__ == '__main__':
    # define a SfSNet
    net = SfSNet().cuda()
    # set to eval mode
    net.eval()
    # load weights
    # net.load_weights_from_pkl('SfSNet-Caffe/weights.pkl')
    net.load_state_dict(torch.load('data/SfSNet.pth'))
    # define a mask generator
    mg = MaskGenerator()

    # get image list
    image_list = glob.glob(M_PIE_DIR)

    for image_name in image_list:
        # read image
        image = cv2.imread(image_name)
        # crop face and generate mask of face
        mask, im = mg.align(image, crop_size=(M, M))
        cv2.imshow('mask', mask*255)
        cv2.imshow('image', im)
        # exit()
        # resize
        im = cv2.resize(im, (M, M))
        # normalize to (0, 1.0)
        im = np.float32(im) / 255.0
        # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.transpose(im, [2, 0, 1])
        im = np.expand_dims(im, 0)

        # get the normal, albedo and light parameter
        normal, albedo, light = net(torch.from_numpy(im).cuda())

        # get numpy array
        n_out = normal.cpu().detach().numpy()
        al_out = albedo.cpu().detach().numpy()
        light_out = light.cpu().detach().numpy()

        # -----------add by wang-------------
        # from [1, 3, 128, 128] to [128, 128, 3]
        n_out = np.squeeze(n_out, 0)
        n_out = np.transpose(n_out, [2, 1, 0])
        # from [1, 3, 128, 128] to [128, 128, 3]
        al_out = np.squeeze(al_out, 0)
        al_out = np.transpose(al_out, [2, 1, 0])
        # from [1, 27] to [27, 1]
        light_out = np.transpose(light_out, [1, 0])
        # print n_out.shape, al_out.shape, light_out.shape
        # -----------end---------------------

        """
        light_out is a 27 dimensional vector. 9 dimension for each channel of
        RGB. For every 9 dimensional, 1st dimension is ambient illumination
        (0th order), next 3 dimension is directional (1st order), next 5
        dimension is 2nd order approximation. You can simply use 27
        dimensional feature vector as lighting representation.
        """

        # transform
        n_out2 = n_out[:, :, (2, 1, 0)]
        # print 'n_out2 shape', n_out2.shape
        n_out2 = cv2.rotate(n_out2, cv2.ROTATE_90_CLOCKWISE)  # imrotate(n_out2,-90)
        n_out2 = np.fliplr(n_out2)
        n_out2 = 2 * n_out2 - 1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2 ** 2, axis=2))  # nr=sqrt(sum(n_out2.^2,3))
        nr = np.expand_dims(nr, axis=2)
        n_out2 = n_out2 / np.repeat(nr, 3, axis=2)
        # print 'nr shape', nr.shape

        al_out2 = cv2.rotate(al_out, cv2.ROTATE_90_CLOCKWISE)
        al_out2 = al_out2[:, :, (2, 1, 0)]
        al_out2 = np.fliplr(al_out2)

        # Note: n_out2, al_out2, light_out is the actual output
        Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)

        diff = (mask // 255).astype(np.float32)
        n_out2 = n_out2 * diff
        al_out2 = al_out2 * diff
        Ishd = Ishd * diff
        Irec = Irec * diff

        # -----------add by wang------------
        # al_out2 = (al_out2 / np.max(al_out2) * 255).astype(dtype=np.uint8)
        # Irec = (Irec / np.max(Irec) * 255).astype(dtype=np.uint8)
        # Ishd = (Ishd / np.max(Ishd) * 255).astype(dtype=np.uint8)

        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)
        al_out2 = cv2.cvtColor(al_out2, cv2.COLOR_RGB2BGR)
        n_out2 = cv2.cvtColor(n_out2, cv2.COLOR_RGB2BGR)
        Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
        # -------------end---------------------

        cv2.imshow("Normal", n_out2)
        cv2.imshow("Albedo", al_out2)
        cv2.imshow("Recon", Irec)
        cv2.imshow("Shading", Ishd)

        cv2.imwrite('result/shading/'+image_name.split('/')[-1], convert(Ishd))
        cv2.imwrite('result/Albedo/'+image_name.split('/')[-1], convert(al_out2))
        cv2.imwrite('result/Irec/'+image_name.split('/')[-1], convert(Irec))
        if cv2.waitKey(0) == 27:
            exit()


if __name__ == '__main__':
    pass
