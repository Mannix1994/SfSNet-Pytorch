# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import numpy as np
import cv2
import torch
from config import M, LANDMARK_PATH, PROJECT_DIR
from src import *


class SfSNetEval:
    def __init__(self, weight_path, landmark_path):
        """
        :param weight_path: weights saved by train.py, not data/SfSNet.pth
        :param landmark_path: face landmark path
        """
        # define a SfSNet
        net = SfSNet()
        # set to eval mode
        net.eval()
        # load weights
        net.load_state_dict(torch.load(weight_path))
        # use cuda
        if torch.cuda.is_available():
            net = net.cuda()
        self.net = net

        # define a mask generator
        self.mg = MaskGenerator(landmark_path)

        # tool layers
        self.normal_layer = NormLayer()
        self.change_form_layer = ChangeFormLayer()
        if torch.cuda.is_available():
            self.shading_layer = ShadingLayer(gpu=True)
        else:
            self.shading_layer = ShadingLayer(gpu=False)

    def _read_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif not isinstance(image, np.ndarray):
            raise RuntimeError('image is not a str or numpy array')
        # crop face and generate mask of face
        mask, im = self.mg.align(image, crop_size=(M, M))
        o_im = im.copy()
        # resize
        im = cv2.resize(im, (M, M))
        # normalize to (0, 1.0)
        im = np.float32(im) / 255.0
        # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.transpose(im, [2, 1, 0])
        im = np.expand_dims(im, 0)

        mask = mask // 255

        im = torch.from_numpy(im)
        if torch.cuda.is_available():
            im = im.cuda()
        return mask, im, o_im

    @staticmethod
    def _get_numpy(tensor):
        n_array = tensor.cpu().detach().numpy()
        n_array = np.squeeze(n_array, 0)
        n_array = np.transpose(n_array, [2, 1, 0])
        return n_array

    def predict(self, image, with_mask=False):
        # compute mask and image
        mask, im, o_im = self._read_image(image)
        # forward net
        Nconv0, Acov0, fc_light = self.net(im)

        # normalize
        normalize = self.normal_layer(Nconv0)
        # change channel of normal
        norch1 = self.change_form_layer(normalize)
        # compute shading
        shading = self.shading_layer(norch1, fc_light)
        # change channel od albedo
        albech2 = self.change_form_layer(Acov0)
        # get recon images
        recon = albech2 * shading

        # -----------add by wang------------
        Ishd = self._get_numpy(shading)  # shading
        al_out2 = self._get_numpy(Acov0)  # albedo
        n_out2 = self._get_numpy(Nconv0)  # normal
        Irec = self._get_numpy(recon)  # reconstructed image

        Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)
        # -------------end---------------------

        if not with_mask:
            return o_im, Irec, n_out2, al_out2, Ishd
        else:
            return o_im * mask, Irec * mask, n_out2 * mask, al_out2 * mask, Ishd * mask[..., 0]


if __name__ == '__main__':
    # get image list
    image_list = glob.glob(os.path.join(PROJECT_DIR, 'Images/*.*'))

    # define sfsnet tool
    ss = SfSNetEval('data/temp_2019.04.10_09.49.20.pth', LANDMARK_PATH)

    for image_name in image_list:
        # read image
        image = cv2.imread(image_name)
        # crop face and generate mask of face
        o_im, Irec, n_out2, al_out2, Ishd = ss.predict(image, False)

        cv2.imshow("image", o_im)
        cv2.imshow("Normal", n_out2)
        cv2.imshow("Albedo", al_out2)
        cv2.imshow("Recon", Irec)
        cv2.imshow("Shading", Ishd)

        # cv2.imwrite('shading.png', convert(Irec))
        if cv2.waitKey(0) == 27:
            exit()


if __name__ == '__main__':
    pass
