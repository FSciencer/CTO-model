
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from samplers import SubsetRandomSampler
import torch.distributed as dist

import numpy as np
import read_filename as rf


class MyDatasets(Dataset):
    def __init__(self, filename_image, filename_label):

        self.image_label_list = []

        f1 = open(filename_image, 'r'); image_path_list = f1.readlines(); f1.close()
        f2 = open(filename_label, 'r'); label_path_list = f2.readlines(); f2.close()
        p1 = [s.rstrip().split() for s in image_path_list]
        p2 = [s.rstrip().split() for s in label_path_list]

        self.image_label_list = [(image_path[0], label_path[0]) for image_path, label_path in zip(p1, p2)]

    def __getitem__(self, index):
        image_label_pair = self.image_label_list[index]

        slice_img = np.load(image_label_pair[0])
        slice_label = np.load(image_label_pair[1])

        slice_img[slice_img < 300] = 300
        slice_img[slice_img > 1000] = 1000
        slice_label[slice_label == 1] = 0
        slice_label[slice_label > 1] = 1

        slice_img = (slice_img - 300) / 700.0

        slice_img = np.expand_dims(slice_img, axis=0)
        slice_label = np.expand_dims(slice_label, axis=0)

        return slice_img, slice_label

    def __len__(self):
        return len(self.image_label_list)


def build_loader(args, config):
    filename_generator = rf.ReadFilename(args.png_path, args.image_path, args.label_path)
    filename_generator()

    dataset_train = MyDatasets('./train_image_filename.txt', './train_label_filename.txt')
    dataset_val = MyDatasets('./validation_image_filename.txt', './validation_label_filename.txt')

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
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE_VAL,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        shuffle=False,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val
