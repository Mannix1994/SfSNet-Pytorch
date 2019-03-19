# coding=utf8
from __future__ import absolute_import, division, print_function

import caffe
import h5py
import numpy as np

if __name__ == '__main__':
    pass

f = h5py.File('wow/SfSNet.caffemodel.h5', 'r')
for group in f.keys():
    # print(group)
    # 根据一级组名获得其下面的组
    group_read = f[group]
    for subgroup in group_read.keys():
        # print('----'+subgroup)
        # 根据一级组和二级组名获取其下面的dataset
        dset_read = f[group + '/' + subgroup]
        # 遍历该子组下所有的dataset
        for dset in dset_read.keys():
            # 获取dataset数据
            dset1 = f[group + '/' + subgroup + '/' + dset]
            data = np.array(dset1)
            print(dset1.name, data.shape)
