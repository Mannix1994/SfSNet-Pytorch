# coding=utf8
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import csv
import warnings


def _convert(src, max_value):
    # find min and max
    _min = np.min(src)
    _max = np.max(src)
    # scale to (0, max_value)
    dst = (src - _min) / (_max - _min + 1e-10)
    dst *= max_value
    return dst


def convert(src, dtype=np.uint8, max_value=255.0):
    # type: (np.ndarray, object, float) -> np.ndarray
    # copy src
    dst = src.copy()
    if src.ndim == 2:
        dst = _convert(dst, max_value)
    elif src.ndim == 3:
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
        light_channel = _convert(dst[0], max_value)
        dst[0, ...] = light_channel
        dst = cv2.cvtColor(dst, cv2.COLOR_LAB2BGR)*255
    else:
        raise RuntimeError("src/utils.py(30): src.ndim should be 2 or 3")
    return dst.astype(dtype)


class Statistic:
    def __init__(self, csv_name, auto_save=False, *keys):
        # CSV文件名字
        self.__csv_name = csv_name
        # 自动保存
        self.__auto_save = auto_save
        # CSV的标题行
        self.__keys = keys
        # 是否有修改的标志位
        self._have_change = False

        if self.__auto_save:
            self.__csv_file = open(csv_name, 'w')
            self.__csv_writer = csv.writer(self.__csv_file)
            self.__csv_writer.writerow(self.__keys)
            self.__step = 0
        else:
            # 记录（一行是一条记录）
            self.__records = []

    def add(self, *values):
        self._have_change = True
        if len(values) != len(self.__keys):
            warnings.warn('the length of values is not equal with keys')
        if self.__auto_save:
            self.__csv_writer.writerow(values)
            self.__step += 1
            # 每调用add10次flush一次，保证效率以及减少数据丢失
            if self.__step == 10:
                self.__csv_file.flush()
                self.__step = 0
        else:
            self.__records.append(values)

    def save(self):
        if self.__auto_save:
            self.__csv_file.close()
        else:
            # 保存记录
            with open(self.__csv_name, 'w') as f:
                writer = csv.writer(f)
                # 写入标题行
                writer.writerow(self.__keys)
                # 写入记录
                for record in self.__records:
                    writer.writerow(record)

        self._have_change = False

    def __del__(self):
        import sys
        if self._have_change:
            sys.stderr.write('Statistic: 还有数据没保存啊！！！\n')
