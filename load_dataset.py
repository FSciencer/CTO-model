#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:09:43 2022

@author: weiwei
@function: 使用 Dataloader 根据文件名列表读取 CTO npy 数据
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from samplers import SubsetRandomSampler
import torch.distributed as dist

import numpy as np
import cv2

import read_filename as rf


# 实现MyDatasets类
class MyDatasets(Dataset):
    def __init__(self, filename_image, filename_label):

        self.image_label_list = []   # 用于存放 tuple-(image, label) 的 list

        # 分别从 image/label 的.txt 中读取到文件名路径
        f1 = open(filename_image, 'r'); image_path_list = f1.readlines(); f1.close()
        f2 = open(filename_label, 'r'); label_path_list = f2.readlines(); f2.close()
        p1 = [s.rstrip().split() for s in image_path_list]  # image_path_list: ['xxxx.tfrecords\n', ...]; p1: ['xxxx.tfrecords', ...]
        p2 = [s.rstrip().split() for s in label_path_list]  # s.rstrip()删除字符串末尾指定字符（默认是字符）

        # 将所有 image-label对 都放入列表，如果要执行多个epoch，可以在这里多复制几遍，然后统一shuffle比较好
        self.image_label_list = [(image_path[0], label_path[0]) for image_path, label_path in zip(p1, p2)]  # 因为zip()操作后，取出来的是list，所以 image_path, label_path 后面加了一个索引

    def __getitem__(self, index):
        image_label_pair = self.image_label_list[index]

        '''
        按 path 读取 img/label (.png 文件)
        path: 每一张 slice 的绝对路径
        '''
        slice_img = np.load(image_label_pair[0])
        slice_label = np.load(image_label_pair[1])

        # pre-process-img: 设定窗宽、窗位
        slice_img[slice_img < 300] = 300
        slice_img[slice_img > 1000] = 1000
        # pre-process-label: 不考虑主动脉(class=1)，冠脉不同分支(class=2/3)作为同一类
        slice_label[slice_label == 1] = 0
        slice_label[slice_label > 1] = 1

        # 归一化
        slice_img = (slice_img - 300) / 700.0

        # transform
        slice_img = np.expand_dims(slice_img, axis=0)
        slice_label = np.expand_dims(slice_label, axis=0)

        return slice_img, slice_label  # image, label

    def __len__(self):
        return len(self.image_label_list)


def build_loader(args, config):
    filename_generator = rf.ReadFilename(args.png_path, args.image_path, args.label_path)
    filename_generator()

    dataset_train = MyDatasets('./train_image_filename.txt', './train_label_filename.txt')
    dataset_val = MyDatasets('./test_image_filename.txt', './test_label_filename.txt')

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 验证/测试数据，按顺序输入测试
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,  # 若想把剩下的不足 batch_size 个数据丢弃，则将 drop_last 设置为True
        # shuffle=True,  # 因为已经使用 sampler 随机打乱，所以不用再设置 shuffle 参数
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE_VAL,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        shuffle=False,  # 验证/测试数据，按顺序输入测试
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val
