# coding=utf8
from __future__ import absolute_import, division, print_function
import h5py
import pickle as pkl


if __name__ == '__main__':
    f = h5py.File('SfSNet.caffemodel.h5', 'r')
    for group_name in f.keys():
        # print(group_name)
        # 根据一级组名获得其下面的组
        name_weights = {}
        group = f[group_name]
        for sub_group_name in group.keys():
            # print('----'+sub_group_name)
            if sub_group_name not in name_weights.keys():
                name_weights[sub_group_name] = {}
            # 根据一级组和二级组名获取其下面的dataset
            # 经过实验，一个dataset对应一层的参数
            dataset = f[group_name + '/' + sub_group_name]
            # 遍历该子组下所有的dataset。
            # print(dataset.keys())
            if len(dataset.keys()) == 1:
                # 如果参数只有一个，则说明是反卷积层,
                # SfSNet整个模型里就只有反卷积层只有一组weight参数
                weight = dataset['0'][()]
                name_weights[sub_group_name]['weight'] = weight

                print('%s:\n\t%s (weight)' % (sub_group_name, weight.shape))
            elif len(dataset.keys()) == 2:
                # 如果参数有两个，则说明是卷积层或者全连接层。
                # 卷积层或者全连接层都有两组参数：weight和bias
                # 权重参数
                weight = dataset['0'][()]
                # print(type(weight))
                # print(weight.shape)
                name_weights[sub_group_name]['weight'] = weight
                # 偏置参数
                bias = dataset['1'][()]
                name_weights[sub_group_name]['bias'] = bias

                print('%s:\n\t%s (weight)' % (sub_group_name, weight.shape))
                print('\t%s (bias)' % str(bias.shape))
            elif len(dataset.keys()) == 3:
                # 如果有三个，则说明是BatchNorm层。
                # BN层共有三个参数，分别是：running_mean、running_var和一个缩放参数。
                running_mean = dataset['0'][()]  # running_mean
                name_weights[sub_group_name]['running_mean'] = running_mean / dataset['2'][()]
                running_var = dataset['1'][()]   # running_var
                name_weights[sub_group_name]['running_var'] = running_var / dataset['2'][()]

                print('%s:\n\t%s (running_var)' % (sub_group_name, running_var.shape), )
                print('\t%s (running_mean)' % str(running_mean.shape))
            elif len(dataset.keys()) == 0:
                # 没有参数
                continue
            else:
                # 如果报错，大家要检查自己模型哈
                raise RuntimeError("还有参数个数超过3个的层，别漏了兄dei！！！\n")

        with open('weights1.pkl', 'wb') as f:
            pkl.dump(name_weights, f, protocol=2)
