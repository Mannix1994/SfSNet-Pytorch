# coding=utf8
from __future__ import absolute_import, division, print_function

import caffe
import h5py
import numpy as np
import pickle as pkl


if __name__ == '__main__':
    f = h5py.File('SfSNet.caffemodel.h5', 'r')
    for group in f.keys():
        # print(group)
        # 根据一级组名获得其下面的组
        name_weights = {}
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
                _, _,  key, index = str(dset1.name).split('/')
                print(key, index)
                if key not in name_weights.keys():
                    name_weights[key] = {}
                if index == '0':
                    name_weights[key]['weight'] = data
                elif index == '1':
                    name_weights[key]['bias'] = data
        with open('weights1.pkl', 'wb') as f:
            pkl.dump(name_weights, f, protocol=2)
