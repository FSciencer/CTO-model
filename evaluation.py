#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 7 19:48:48 2022

@author: weiwei
@function: 预测过程的功能函数集合
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def dice_similarity_coefficient(prediction, target):
    # prediction = np.reshape(prediction, [-1, 1])
    # target = np.reshape(target, [-1, 1])
    prediction = prediction.flatten()
    target = target.flatten()
    smooth = 1e-5
    intersect = np.sum(prediction * target)
    z_sum = np.sum(prediction)
    y_sum = np.sum(target)
    dice_metric = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice_metric


def save_image(input_, predictions, label, save_dir='.', max_=1, min_=0):

    f, axes = plt.subplots(1, 3, figsize=(30, 20))

    axes[0].imshow(input_, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1].imshow(predictions, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[2].imshow(label, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[0].title.set_text('input_image')
    axes[1].title.set_text('prediction')
    axes[2].title.set_text('label')

    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()
