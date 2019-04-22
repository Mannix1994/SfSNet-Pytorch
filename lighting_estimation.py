# coding=utf8
from __future__ import print_function
import cv2
import numpy as np
import time
import csv


def draw_arrow(image, magnitude, angle, magnitude_threshold=1.0, length=10):
    # _image = image.copy()
    _image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    angle = angle / 180.0 * np.pi
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            magni = magnitude[i, j]
            ang = angle[i, j]
            if magni < magnitude_threshold:
                continue
            diff_i = int(np.round(np.sin(ang) * length))
            diff_j = int(np.round(np.cos(ang) * length))
            cv2.line(_image, (j, i), (j + diff_j, i + diff_i), (0, 255, 0))
            p_i = np.max((0, i + diff_i))
            p_i = np.min((_image.shape[0] - 1, p_i))
            p_j = np.max((0, j + diff_j))
            p_j = np.min((_image.shape[1] - 1, p_j))
            _image[p_i, p_j] = (0, 0, 255)
    return _image


def which_direction(image, mask, magnitude_threshold=1.0, show_arrow=False):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 转换为浮点类型
    gray = np.float32(image)
    # define horizontal filter kernel
    h_kernel = np.array((-1, -1, 0, 1, 1)).reshape(1, -1)
    # define vertical filter kernel
    v_kernel = h_kernel.T.copy()
    # filter horizontally
    h_conv = cv2.filter2D(gray, -1, kernel=h_kernel)
    # filter vertical
    v_conv = cv2.filter2D(gray, -1, kernel=v_kernel)
    # compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(h_conv, v_conv, angleInDegrees=True)
    if mask is not None:
        _mask = mask[:, :, 0]
        _mask = _mask
        # remove the un-masked area
        magnitude *= _mask
        angle *= _mask
    # draw some arrow
    if show_arrow:
        im = draw_arrow(image, magnitude, angle, magnitude_threshold)
        cv2.namedWindow('arrow', cv2.WINDOW_NORMAL)
        cv2.imshow('arrow', im)
        cv2.waitKey(50)
    # set angle[i,j]=0 if magnitude[i, j] < magnitude_threshold
    angle = angle * np.int32(magnitude > magnitude_threshold)
    # count the angle's direction
    # please see doc/light_estimation_分区.png
    right_down_1 = np.sum(np.int32((angle > 0) & (angle < 45)))
    right_down_2 = np.sum(np.int32((angle >= 45) & (angle < 90)))
    left_down_3 = np.sum(np.int32((angle >= 90) & (angle < 135)))
    left_down_4 = np.sum(np.int32((angle >= 135) & (angle < 180)))
    left_up_5 = np.sum(np.int32((angle >= 180) & (angle < 225)))
    left_up_6 = np.sum(np.int32((angle >= 225) & (angle < 270)))
    right_up_7 = np.sum(np.int32((angle >= 270) & (angle < 315)))
    right_up_8 = np.sum(np.int32((angle >= 315) & (angle < 360)))

    angle_in_range = [[1, right_down_1],
                      [2, right_down_2],
                      [3, left_down_3],
                      [4, left_down_4],
                      [5, left_up_5],
                      [6, left_up_6],
                      [7, right_up_7],
                      [8, right_up_8]]
    # 判断属于那个方向
    direction = _which_direction(angle_in_range)

    return direction, angle_in_range


def _which_direction(angle_in_range, debug=False):
    """
    please see doc/light_estimation_方向.png
    这个函数判断逻辑，emmm，有点复杂，很多阈值都是经验值，判断规则也是经验。
    所以用这个函数的时候需要根据自己的数据集调整阈值。
    """
    # 找到最大值
    _max = max(angle_in_range, key=lambda x: x[1])[1]
    _max = float(_max)
    # 每个角度范围的值都除以最大值
    _avg_angle_in_range = [(r, round(l / _max, 2)) for r, l in angle_in_range]
    # 排序
    s = sorted(_avg_angle_in_range, key=lambda x: x[1], reverse=True)

    if debug:
        print('s=', s)
    # 前四个的占比都超过0.65
    if s[3][1] > 0.65:
        # 排序
        ss = sorted(s[0:4], key=lambda x: x[0])
        if debug:
            print('ss=', ss)
        # 连续的三种特殊情况
        zheyelianxu = [[1, 6, 7, 8], [1, 2, 7, 8], [1, 2, 3, 8]]
        xuhao = [sss[0] for sss in ss]
        if xuhao in zheyelianxu:
            if xuhao == [1, 6, 7, 8]:
                return 7
            elif xuhao == [1, 2, 7, 8]:
                return 8
            else:
                return 1
        else:
            # 连续，返回中间的值
            if xuhao[3] - xuhao[0] == 3:
                return (xuhao[1] + xuhao[2]) / 2.0
            else:
                # 如果不连续，那么认为是均匀光照
                return -1
    # 如果前两个的占比之差超过0.16，直接选第一个
    if (s[0][1] - s[1][1]) > 0.16:
        return s[0][0]
    # 如果第二个和第三个的差大于0.1，则取前两个的平均值
    elif (s[1][1] - s[2][1]) > 0.1:
        # 序号是1和8，是连续的，直接返回8
        if (s[1][0] == 8 and s[0][0] == 1) or (s[1][0] == 1 and s[0][0] == 8):
            return 8
        # 序号之差大于2，认为是均匀光照
        elif abs(s[1][0] - s[0][0]) > 2:
            return -1
        # 否则取平均值
        else:
            return (s[1][0] + s[0][0]) / 2.0
    # 第三个和第四个的差大于0.1，则认为前三个连续
    elif (s[2][1] - s[3][1]) > 0.1:
        # 排序
        ss = sorted(s[0:3], key=lambda x: x[0])
        if debug:
            print('ss=', ss)
        # 计算两两序号之差
        _1_0 = ss[1][0] - ss[0][0]
        _2_1 = ss[2][0] - ss[1][0]
        _2_0 = ss[2][0] - ss[0][0]
        # 序号是128和178是连续的
        if (np.array(ss)[:, 0] == np.array([1, 2, 8])).all():
            if debug:
                print('128--------------------128---------')
            return ss[0][0]
        elif (np.array(ss)[:, 0] == np.array([1, 7, 8])).all():
            if debug:
                print('178--------------------178---------')
            return ss[2][0]
        # 序号之差大于1，说明这三个序号不连续，认为是平均光照
        if _1_0 > 1 or _2_1 > 1:
            return -1
        # 三个序号连续，取中间的序号
        elif _1_0 == 1 and _2_1 == 1:
            return ss[1][0]
        # 没啥用
        else:
            return -1
    # 有四个连续，则直接判断为均匀光照
    else:
        return -1


class Statistic:
    def __init__(self, csv_name, auto_save=False, *keys):
        # CSV文件名字
        self._csv_name = csv_name
        # 自动保存
        self._auto_save = auto_save
        # CSV的标题行
        self._keys = list(keys)
        # 记录（一行是一条记录）
        self._records = {}
        # 是否保存的标志位
        self._have_change = False

    def add(self, face_id, key):
        # 设置标志
        self._have_change = True
        # 还没有face_id指定的人
        if face_id not in self._records.keys():
            # 如果设置了自动保存，保存一下之前统计的信息
            if self._auto_save:
                self.save()
            # 创建用于统计该人信息的字典
            self._records[face_id] = {}
            # 创建该人的所有key
            for k in self._keys:
                self._records[face_id][k] = 0
        # 找到对应的人的对应key，使其值加一
        self._records[face_id][key] += 1

    def save(self):
        # 保存记录
        with open(self._csv_name, 'w') as f:
            writer = csv.writer(f)
            # 写入标题行
            writer.writerow(['index', 'id', ] + self._keys)
            # 写入记录
            index = 1
            for fid, key_val in self._records.items():
                # print (fid, key_val)
                writer.writerow([index, fid, ] + [key_val[k] for k in self._keys])
                index += 1

        self._have_change = False

    def __del__(self):
        import sys
        if self._have_change:
            sys.stderr.write('Statistic: 还有数据没保存啊！！！\n')


gray_level_keys = ['<70', '70-115', '115-160', '160-205', '205-255']


def gray_level(shading, mask):
    """
    按照shading的像素平均值，把图像亮度分成五个等级。
    :param shading:
    :param mask:
    :return:
    """
    # 计算像素总数
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, :, 0] / 255
        pixel_count = np.sum(mask)
    else:
        pixel_count = shading.size
    # 计算shading的像素总值
    shading_count = np.sum(shading)
    # 计算每个像素的平均值
    avg_pixel_val = shading_count / pixel_count
    # 判断等级
    if avg_pixel_val < 70:
        level = gray_level_keys[0]
    elif avg_pixel_val < 115:
        level = gray_level_keys[1]
    elif avg_pixel_val < 160:
        level = gray_level_keys[2]
    elif avg_pixel_val < 205:
        level = gray_level_keys[3]
    elif avg_pixel_val < 255:
        level = gray_level_keys[4]
    else:
        level = gray_level_keys[0]
    return avg_pixel_val, level


if __name__ == '__main__':
    s = Statistic('123.csv', False, 'left', 'right', 'direct')
    s.add(1, 'left')
    s.add(1, 'right')
    s.add(1, 'right')
    s.add(4, 'direct')
    s.add(1, 'direct')
    s.add(2, 'left')
    s.add(2, 'right')
    s.add(2, 'right')
    s.add(3, 'direct')
    s.add(2, 'direct')
    s.save()

if __name__ == '__main__':
    image = cv2.imread('SfSNet/shading.png', cv2.IMREAD_GRAYSCALE)
    print(which_direction(image, None, 1, True))
