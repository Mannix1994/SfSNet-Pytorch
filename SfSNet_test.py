# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import cv2
import numpy as np
import pickle as pkl
from torch.nn import init, Parameter
from src.models.model import SfSNet
from src.functions import create_shading_recon
from src.mask import MaskGenerator
from config import M, LANDMARK_PATH


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def save(path, arr):
    if isinstance(arr, np.ndarray):
        np.save(path+'.torch.npy', arr)
    else:
        np.save(path+'.torch.npy', arr.detach().numpy())


if __name__ == '__main__':
    net = SfSNet(bn_affine=True)
    net.eval()
    net.load_weights_from_pkl('wow/weights.pkl')

    mg = MaskGenerator(LANDMARK_PATH)
    image = cv2.imread('1.png_face.png')
    mask, im = mg.align(image, crop_size=(M, M))
    im = cv2.resize(im, (M, M))
    im = np.float32(im) / 255.0
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, [2, 0, 1])  # from (128, 128, 3) to (1, 3, 128, 128)
    im = np.expand_dims(im, 0)
    print(np.min(im), np.max(im))

    # conv1 = net(torch.from_numpy(im))

    # print(conv1[2].detach().numpy())
    # exit()

    # np.save('data/nsum5.torch.npy', conv1[0].detach().numpy())
    # np.save('data/asum5.torch.npy', conv1[1].detach().numpy())
    # np.save('data/lconcat1.torch.npy', conv1[2].detach().numpy())
    #
    # exit()
    #
    # npp = conv1.detach().numpy()
    # print(npp.shape)
    # print(npp[0, :, 0, 0])
    # duibi = np.load('data/data.npy')
    # indices = np.arange(0, 256, 1)
    # print(np.sum(np.abs(npp-duibi)))
    # exit()

    normal, albedo, light = net(torch.from_numpy(im))

    n_out=normal.detach().numpy()
    al_out=albedo.detach().numpy()
    light_out = light.detach().numpy()

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

    diff = (mask // 255)
    n_out2 = n_out2 * diff
    al_out2 = al_out2 * diff
    Ishd = Ishd * diff
    Irec = Irec * diff

    save('data/normal', n_out2)
    save('data/albedo', al_out2)
    save('data/light', light_out)
    save('data/Irec', Irec)
    save('data/Ishd', Ishd)
    print(diff.dtype)

    # -----------add by wang------------
    Ishd = np.float32(Ishd)
    Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)

    print(np.min(al_out2))

    al_out2 = (al_out2 / np.max(al_out2) * 255).astype(dtype=np.uint8)
    Irec = (Irec / np.max(Irec) * 255).astype(dtype=np.uint8)
    Ishd = (Ishd / np.max(Ishd) * 255).astype(dtype=np.uint8)

    al_out2 = cv2.cvtColor(al_out2, cv2.COLOR_RGB2BGR)
    n_out2 = cv2.cvtColor(n_out2, cv2.COLOR_RGB2BGR)
    Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
    # -------------end---------------------
    cv2.imshow("Normal", n_out2)
    cv2.imshow("Albedo", al_out2)
    cv2.imshow("Recon", Irec)
    cv2.imshow("Shading", Ishd)
    cv2.waitKey(0)


if __name__ == '__main__':
    pass
