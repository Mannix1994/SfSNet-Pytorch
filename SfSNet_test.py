# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import numpy as np
import cv2
import torch
from config import M, LANDMARK_PATH, PROJECT_DIR
from src import create_shading_recon, convert
from src import MaskGenerator
from src import SfSNet

if __name__ == '__main__':
    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    # net.load_weights_from_pkl('SfSNet-Caffe/weights.pkl')
    net.load_state_dict(torch.load('weights/SfSNet.pth'))
    # define a mask generator
    mg = MaskGenerator(LANDMARK_PATH)

    # get image list
    image_list = glob.glob(os.path.join(PROJECT_DIR, 'Images/*.*'))

    for image_name in image_list:
        # read image
        image = cv2.imread(image_name)
        # crop face and generate mask of face
        aligned, mask, im, _ = mg.align(image, size=(M, M))[0]
        mask = mask // 255
        cv2.imshow('image', im*mask)
        # resize
        im = cv2.resize(im, (M, M))
        # normalize to (0, 1.0)
        im = np.float32(im) / 255.0
        # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.transpose(im, [2, 0, 1])
        im = np.expand_dims(im, 0)

        # get the normal, albedo and light parameter
        normal, albedo, light = net(torch.from_numpy(im))

        # get numpy array
        n_out = normal.detach().numpy()
        al_out = albedo.detach().numpy()
        light_out = light.detach().numpy()

        # -----------add by wang-------------
        # from [1, 3, 128, 128] to [128, 128, 3]
        n_out = np.squeeze(n_out, 0)
        n_out = np.transpose(n_out, [1, 2, 0])
        # from [1, 3, 128, 128] to [128, 128, 3]
        al_out = np.squeeze(al_out, 0)
        al_out = np.transpose(al_out, [1, 2, 0])
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
        # n_out2 = n_out[:, :, (2, 1, 0)]  # BGR to RGB
        n_out2 = cv2.cvtColor(n_out, cv2.COLOR_BGR2RGB)  # BGR to RGB
        # print 'n_out2 shape', n_out2.shape
        n_out2 = 2 * n_out2 - 1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2 ** 2, axis=2, keepdims=True))  # nr=sqrt(sum(n_out2.^2,3))
        # nr = np.expand_dims(nr, axis=2)
        n_out2 = n_out2 / np.repeat(nr, 3, axis=2)
        # print('nr shape', nr.shape)

        # al_out2 = al_out[:, :, (2, 1, 0)]  # BGR to RGB
        al_out2 = cv2.cvtColor(al_out, cv2.COLOR_BGR2RGB)  # BGR to RGB

        # Note: n_out2, al_out2, light_out is the actual output
        Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)
        # print(Irec.dtype, Ishd.dtype)

        n_out2 = n_out2 * mask
        al_out2 = al_out2 * mask
        Ishd = Ishd * mask
        Irec = Irec * mask

        # -----------add by wang------------

        al_out2 = cv2.cvtColor(al_out2, cv2.COLOR_RGB2BGR)  # RGB to BGR
        n_out2 = cv2.cvtColor(n_out2, cv2.COLOR_RGB2BGR)
        Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2BGR)

        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)
        # -------------end---------------------

        cv2.imshow("Normal", n_out2)
        cv2.imshow("Albedo", al_out2)
        cv2.imshow("Recon", Irec)
        cv2.imshow("Shading", Ishd)

        # cv2.imwrite('shading.png', convert(Irec))
        if cv2.waitKey(0) == 27:
            exit()


if __name__ == '__main__':
    pass
