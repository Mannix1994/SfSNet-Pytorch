# coding=utf8

from __future__ import absolute_import, division, print_function

import dlib
import cv2
import sys
import numpy as np
from src.functions import create_mask_fiducial
from config import LANDMARK_PATH


class MaskGenerator:
    def __init__(self, landmarks_path=LANDMARK_PATH):
        """
        :param landmarks_path: the path of pretrained key points weight,
        it could be download from:
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        import os
        if not os.path.exists(LANDMARK_PATH):
            raise RuntimeError('face landmark file is not exist. please download if from: \n'
                               'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 '
                               'and uncompress it.')
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(landmarks_path)

    def get_masked_face(self, image):
        mask, img = self.align(image)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(50)
        return mask & img

    def bounding_boxes(self, image):
        # convert to gray image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # get rect contains face
        face_rects = self._detector(gray_image, 0)
        return face_rects

    def align(self, image, crop_size=(240, 240), scale=4, show_landmarks=False, return_none=False):
        """
        :param show_landmarks:
        :param scale:
        :param crop_size:
        :type image: np.ndarray
        :param image: BGR face image
        :return: a face image with mask
        https://blog.csdn.net/qq_39438636/article/details/79304130
        """
        # get rectangles which contains face
        face_rects = self.bounding_boxes(image)
        for i in range(len(face_rects)):
            # get 68 landmarks of face
            landmarks = np.array([[p.x, p.y] for p in self._predictor(image, face_rects[i]).parts()])
            # show landmarks
            if show_landmarks:
                self.show_landmarks(image, landmarks)
            # create mask using landmarks
            mask = create_mask_fiducial(landmarks.T, image)
            # warp and crop image
            mask, image = self._warp_and_crop_face(image, mask, landmarks, crop_size, scale)
            # # detect rectangle
            # rect = self._detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0)
            # cv2.rectangle(mask, (rect[i].left(), rect[i].top()),
            #               (rect[i].right(), rect[i].bottom()), (255, 255, 0))
            return mask, image, True
        else:
            sys.stderr.write("%s: Can't detect face in image\n" % __file__)
            image = cv2.resize(image, crop_size)
            if return_none:
                return None, image, False
            else:
                return np.ones(image.shape, dtype=image.dtype)*255, image, False

    def _warp_and_crop_face(self, image, mask, landmarks, crop_size, scale):
        """
        :param scale:
        :param image:
        :type image
        :param landmarks:
        :type landmarks np.ndarray
        :param crop_size: tuple
        :return: warp_and_crop_face
        """
        # landmarks.shape = (68, 2)
        landmarks = np.array(landmarks)
        # compute rotate angle, r_angle=arctan((y1-y2)/(x1-x2))
        # landmarks[36]: corner of left eye
        # landmarks[42]: corner of right eye
        r_angle = np.arctan((landmarks[36][1]-landmarks[42][1]) /
                            (landmarks[36][0]-landmarks[42][0]))
        r_angle = 180*r_angle/np.pi
        # get rotation matrix
        rot_mat = cv2.getRotationMatrix2D(tuple(landmarks[2]), r_angle, scale=1)

        # rotate image and mask
        ims = image.shape[0:2]
        rotated_image = cv2.warpAffine(image, rot_mat, dsize=(ims[0]*2, ims[1]*2))
        rotated_mask = cv2.warpAffine(mask, rot_mat, dsize=(ims[0]*2, ims[1]*2))

        # crop image and mask
        cropped_image = self._crop_image(rotated_image, landmarks, scale)
        cropped_mask = self._crop_image(rotated_mask, landmarks, scale)

        # resize image and mask
        resize_image = cv2.resize(cropped_image, crop_size)
        resize_mask = cv2.resize(cropped_mask, crop_size)

        return resize_mask, resize_image

    def _crop_image(self, image, landmarks, scale):
        # compute the distance between left eye(landmarks[36])
        # and left mouth point[landmarks[48]]
        distance = np.sqrt(np.sum(np.square(landmarks[36]-landmarks[48])))
        size = distance * scale
        # compute row_start, row_end, col_start, col_end
        landmark_nose = landmarks[30]
        row_start = int(landmark_nose[1]-size/2.0)
        row_end = int(landmark_nose[1]+size/2.0)
        col_start = int(landmark_nose[0]-size/2.0)
        col_end = int(landmark_nose[0]+size/2.0)
        # print('*' * 10)
        # print(row_start, row_end, col_start, col_end)
        # make range valid
        if row_start < 0:
            row_end += abs(row_start)
            row_start = 0
        if col_start < 0:
            col_end += abs(col_start)
            col_start = 0
        row_end = min(row_end, image.shape[0])
        col_end = min(col_end, image.shape[1])
        # print(row_start, row_end, col_start, col_end)
        # print('*' * 10)
        # crop image
        cropped_image = image[row_start:row_end, col_start:col_end]
        rows, cols, _ = cropped_image.shape
        _min = np.min((rows, cols))
        if _min < cols:
            # rows is smaller than cols
            padding = np.zeros(shape=(cols-_min, cols, 3), dtype=cropped_image.dtype)
            cropped_image = np.vstack((cropped_image, padding))
        elif _min < rows:
            # cols is smaller than rows
            padding = np.zeros(shape=(rows, rows - _min, 3), dtype=cropped_image.dtype)
            cropped_image = np.hstack((cropped_image, padding))
        return cropped_image

    def show_landmarks(self, image, landmarks):
        im = image.copy()
        for i, landmark in enumerate(landmarks):
            cv2.circle(im, tuple(landmark), 3, (0, 0, 255))
            cv2.putText(im, str(i), tuple(landmark), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 0))
        cv2.imshow("landmarks", im)
        cv2.waitKey(50)


if __name__ == '__main__':
    from config import LANDMARK_PATH
    image = cv2.imread('Images/4.png_face.png')
    mask_gen = MaskGenerator(LANDMARK_PATH)
    mah= mask_gen.get_masked_face(image)
    cv2.imshow('123', mah)
    cv2.waitKey(0)
