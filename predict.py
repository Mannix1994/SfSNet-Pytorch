# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import cv2
import torch
import argparse
from config import LANDMARK_PATH, PROJECT_DIR
from src import *


def find_newest_file(directory, suffix='.pth'):
    lists = os.listdir(directory)
    lists = [i for i in lists if i.endswith(suffix)]
    lists.sort(key=lambda f_name: os.path.getmtime(os.path.join(directory, f_name)))
    if len(lists) > 0:
        return os.path.join(directory, lists[-1])
    else:
        raise RuntimeError("No *{} file in directory: {}".format(suffix, directory))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='The path to weights')
    parser.add_argument('-g', '--gpu', action='store_true', help='whether using gpu or not')
    args = parser.parse_args()

    print(args)
    if args.weights is None:
        args.weights = find_newest_file('weights')

    print("\nParameter weights is not defined. "
          "predict.py will use the newest weight file: " + args.weights)

    # get image list
    image_list = glob.glob(os.path.join(PROJECT_DIR, 'Images/*.*'))

    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    net.load_state_dict(torch.load(args.weights))
    # define sfsnet tool
    ss = SfSNetEval(net, LANDMARK_PATH, args.gpu)

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
