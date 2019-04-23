# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import cv2
import torch
import argparse
from config import LANDMARK_PATH, PROJECT_DIR
from lighting_estimation import which_direction
from src import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='The path to weights')
    parser.add_argument('-g', '--gpu', action='store_true', help='whether using gpu or not')
    args = parser.parse_args()

    # get image list
    image_list = glob.glob(os.path.join(PROJECT_DIR, '周/*.*'))

    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    net.load_state_dict(torch.load('weights/SfSNet.pth'))
    # define sfsnet tool
    ss = SfSNetEval(net, LANDMARK_PATH, args.gpu)

    sta = Statistic('data/周.csv', '图片', '方向')
    for image_name in image_list:
        # read image
        image = cv2.imread(image_name)
        # crop face and generate mask of face
        o_im, face, Irec, n_out2, al_out2, Ishd, mask, _, _ = ss.predict(image, True)

        cv2.imshow("image", o_im)
        cv2.imshow("Normal", convert(n_out2))
        cv2.imshow("Albedo", convert(al_out2))
        cv2.imshow("Recon", convert(Irec))

        shading = convert(Ishd)
        shading = cv2.cvtColor(shading, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Shading", shading)

        print(shading.shape)
        direction, angle_count = which_direction(shading, mask, magnitude_threshold=10)
        angle_count = sorted(angle_count, key=lambda x: x[1], reverse=True)
        print(direction, angle_count)
        sta.add(image_name.split('/')[-1], direction)

        # cv2.imwrite('shading.png', convert(Irec))
        if cv2.waitKey(0) == 27:
            sta.save()
            exit()
    sta.save()


if __name__ == '__main__':
    pass
