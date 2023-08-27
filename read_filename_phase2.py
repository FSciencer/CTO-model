
import os
from glob import glob


class ReadFilename(object):
    def __init__(self, data_path, image_path, mask_path, label_path, is_unpair=False):
        # path
        self.data_path = data_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.label_path = label_path

        self.is_unpair = is_unpair

    def __call__(self):

        patient_dir_image = os.path.join(self.data_path, self.image_path)
        patient_dir_mask = os.path.join(self.data_path, self.mask_path)
        patient_dir_label = os.path.join(self.data_path, self.label_path)

        patient_dir_image_train = os.path.join(patient_dir_image, "training_set")
        patient_dir_mask_train = os.path.join(patient_dir_mask, "training_set")
        patient_dir_label_train = os.path.join(patient_dir_label, "training_set")

        patient_dir_image_test = os.path.join(patient_dir_image, "validation_set")
        patient_dir_mask_test = os.path.join(patient_dir_mask, "validation_set")
        patient_dir_label_test = os.path.join(patient_dir_label, "validation_set")

        patient_image_train = [d for d in glob(patient_dir_image_train + '/*')]
        patient_mask_train = [d for d in glob(patient_dir_mask_train + '/*')]
        patient_label_train = [d for d in glob(patient_dir_label_train + '/*')]
        patient_image_test = [d for d in glob(patient_dir_image_test + '/*')]
        patient_mask_test = [d for d in glob(patient_dir_mask_test + '/*')]
        patient_label_test = [d for d in glob(patient_dir_label_test + '/*')]

        patient_image_train.sort()
        patient_mask_train.sort()
        patient_label_train.sort()
        patient_image_test.sort()
        patient_mask_test.sort()
        patient_label_test.sort()

        train_image_filename, train_mask_filename, train_label_filename = self.get_file_list(patient_image_train, patient_mask_train, patient_label_train)
        test_image_filename, test_mask_filename, test_label_filename = self.get_file_list(patient_image_test, patient_mask_test, patient_label_test)

        f1 = open('./train_image_filename.txt', 'w'); f1.writelines(fn + '\n' for fn in train_image_filename); f1.close()
        f2 = open('./train_mask_filename.txt', 'w'); f2.writelines(fn + '\n' for fn in train_mask_filename); f2.close()
        f3 = open('./train_label_filename.txt', 'w'); f3.writelines(fn + '\n' for fn in train_label_filename); f3.close()
        f4 = open('./validation_image_filename.txt', 'w'); f4.writelines(fn + '\n' for fn in test_image_filename); f4.close()
        f5 = open('./validation_mask_filename.txt', 'w'); f5.writelines(fn + '\n' for fn in test_mask_filename); f5.close()
        f6 = open('./validation_label_filename.txt', 'w'); f6.writelines(fn + '\n' for fn in test_label_filename); f6.close()
        print('filename list save to .txt complete !!!')

    def get_file_list(self, list1, list2, list3):
        image_slice_name_list, mask_slice_name_list, label_slice_name_list = [], [], []
        for patient_name_image, patient_name_mask, patient_name_label in zip(list1, list2, list3):
            image_slice_path = glob(os.path.join(patient_name_image, '*'))
            mask_slice_path = glob(os.path.join(patient_name_mask, '*'))
            label_slice_path = glob(os.path.join(patient_name_label, '*'))
            image_slice_path.sort()
            mask_slice_path.sort()
            label_slice_path.sort()

            image_slice_name_list.extend(image_slice_path)
            mask_slice_name_list.extend(mask_slice_path)
            label_slice_name_list.extend(label_slice_path)

        return image_slice_name_list, mask_slice_name_list, label_slice_name_list
