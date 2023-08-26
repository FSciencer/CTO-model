#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:06:03 2023

@author: weiwei
@function: 3D patch ResNet 做冠脉分割
           3D patch ResNet 做CTO特征提取
"""

import os
import argparse
from config import get_config
from model import Patch_UCTNet


def parse_option():
    parser = argparse.ArgumentParser(description='training and evaluation script', add_help=False)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')

    # -------------------------------------
    # Data setting
    parser.add_argument('--png_path', dest='png_path', default='/dataset/CTO/', help='.dicom file directory')
    parser.add_argument('--image_path', dest='image_path', default='cta_resampling_npy', help='CT image folder name')
    parser.add_argument('--label_path', dest='label_path', default='coronary_resampling_npy', help='coronary label folder name')

    # Directory saving setting
    parser.add_argument('--logger_dir', dest='logger_dir', default='./logger', help='print information in console')

    # image info
    parser.add_argument('--img_size', dest='img_size', type=int,  default=512, help='image whole size, h=w')
    parser.add_argument('--img_vmax', dest='img_vmax', type=int, default=255, help='max value in image')
    parser.add_argument('--img_vmin', dest='img_vmin', type=int, default=0,  help='max value in image')

    # easy config modification
    parser.add_argument('--num_classes', type=int, default=1, help="class num of network output")
    parser.add_argument('--epochs', type=int, default=100, help="total training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="batch size for single GPU")
    parser.add_argument('--batch-size_val', type=int, default=32, help="batch size of validation for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, '
                                                                                                       'full: cache all data, '
                                                                                                       'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder, the full path is <output>/<model_name> (default: output)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument('--local_rank', type=int, default=0, required=False, help='local rank for DistributedDataParallel, GPU device id that code will be operating on')

    # -------------------------------------
    args, unknown = parser.parse_known_args()
    config = get_config(args)

    return args, config


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args, config = parse_option()
    Patch_UCTNet(args, config)
