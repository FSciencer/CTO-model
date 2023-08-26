#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:53:06 2022

@author: weiwei
@function: 从硬盘读取文件名列表
"""

import os
from glob import glob
from random import shuffle


class ReadFilename(object):
    def __init__(self, png_path, image_path, label_path, is_unpair=False):
        # path
        self.png_path = png_path
        self.image_path = image_path
        self.label_path = label_path

        self.is_unpair = is_unpair

    def __call__(self):

        # tfrecords dir name
        patient_dir_image = os.path.join(self.png_path, self.image_path)
        patient_dir_label = os.path.join(self.png_path, self.label_path)
        # patients name absolutely path
        patient_dir_image_train = os.path.join(patient_dir_image, "trainset")
        patient_dir_label_train = os.path.join(patient_dir_label, "trainset")
        patient_dir_image_test = os.path.join(patient_dir_image, "testset")
        patient_dir_label_test = os.path.join(patient_dir_label, "testset")
        patient_image_train = [d for d in glob(patient_dir_image_train + '/*')]
        patient_label_train = [d for d in glob(patient_dir_label_train + '/*')]
        patient_image_test = [d for d in glob(patient_dir_image_test + '/*')]
        patient_label_test = [d for d in glob(patient_dir_label_test + '/*')]
        patient_image_train.sort()
        patient_label_train.sort()
        patient_image_test.sort()
        patient_label_test.sort()

        train_image_filename, train_label_filename = self.get_file_list(patient_image_train, patient_label_train)
        test_image_filename, test_label_filename = self.get_file_list(patient_image_test, patient_label_test)

        f1 = open('./train_image_filename.txt', 'w'); f1.writelines(fn + '\n' for fn in train_image_filename); f1.close()
        f2 = open('./train_label_filename.txt', 'w'); f2.writelines(fn + '\n' for fn in train_label_filename); f2.close()
        f3 = open('./test_image_filename.txt', 'w'); f3.writelines(fn + '\n' for fn in test_image_filename); f3.close()
        f4 = open('./test_label_filename.txt', 'w'); f4.writelines(fn + '\n' for fn in test_label_filename); f4.close()
        print('filename list save to .txt complete !!!')

    # generate filename list，用于生成文件名列表
    def get_file_list(self, list1, list2):
        image_slice_name_list, label_slice_name_list = [], []
        for patient_name_image, patient_name_label in zip(list1, list2):
            image_slice_path = glob(os.path.join(patient_name_image, '*'))
            label_slice_path = glob(os.path.join(patient_name_label, '*'))
            image_slice_path.sort()
            label_slice_path.sort()

            image_slice_name_list.extend(image_slice_path)
            label_slice_name_list.extend(label_slice_path)

        return image_slice_name_list, label_slice_name_list
