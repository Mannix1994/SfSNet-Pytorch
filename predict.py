# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import cv2
import torch
import argparse
from config import LANDMARK_PATH, PROJECT_DIR
from src import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='The path to weights',
                        default='data/weights_2019.04.19_19.00.10.pth')
    args = parser.parse_args()

    # get image list
    image_list = glob.glob(os.path.join(PROJECT_DIR, 'Images/*.*'))

    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    net.load_state_dict(torch.load(args.weights))
    # define sfsnet tool
    ss = SfSNetEval(net, LANDMARK_PATH)

    for image_name in image_list:
        # read image
        image = cv2.imread(image_name)
        # crop face and generate mask of face
        o_im, face, Irec, n_out2, al_out2, Ishd, _, _, _ = ss.predict(image, False)

        cv2.imshow("image", o_im)
        cv2.imshow("Normal", convert(n_out2))
        cv2.imshow("Albedo", convert(al_out2))
        cv2.imshow("Recon", convert(Irec))
        cv2.imshow("Shading", convert(Ishd))

        # cv2.imwrite('shading.png', convert(Irec))
        if cv2.waitKey(0) == 27:
            exit()


if __name__ == '__main__':
    pass
