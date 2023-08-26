
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

from apex import amp
from timm.utils import AverageMeter
from tqdm import tqdm

from STformer import SwinTransformer
from optimizer import build_optimizer
from logger import create_logger
from lr_scheduler import build_scheduler
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

import load_dataset as ld


class Patch_UCTNet(object):
    def __init__(self, args, config):

        # save directory
        dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

        os.makedirs(config.OUTPUT, exist_ok=True)
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        self.dataset_train, self.dataset_val, self.data_loader_train, self.data_loader_val = ld.build_loader(args, config)

        model = SwinTransformer()
        model.cuda()
        optimizer = build_optimizer(config, model)
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')  # opt_level: ('O0', 'O1', 'O2')

        # loss function
        ce_loss = nn.CrossEntropyLoss()

        # 寻找最新的检查点
        if config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(config.OUTPUT)
            if resume_file:
                if config.MODEL.RESUME:
                    logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
                config.defrost()
                config.MODEL.RESUME = resume_file
                config.freeze()
                logger.info(f'auto resuming from {resume_file}')
            else:
                logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

        # 从最新的检查点自动恢复模型参数
        if config.MODEL.RESUME:
            load_checkpoint(config, model, optimizer, logger)
            if config.EVAL_MODE:
                return

        logger.info("Start training")
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):  # 或者换成：for epoch in (iterator=tqdm(range(config.TRAIN.EPOCHS), ncols=10)):
            self.data_loader_train.sampler.set_epoch(epoch)

            self.train_one_epoch(config, model, ce_loss, self.data_loader_train, optimizer, epoch, logger)

            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model, optimizer, logger)

    def train_one_epoch(self, config, model, ce_loss, data_loader, optimizer, epoch, logger):
        model.train()

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()

        end = time.time()

        for idx, (samples, targets) in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            samples = samples.to(torch.float32)
            targets = targets.to(torch.float32)

            ###########################################################
            outputs = model(samples)
            loss = ce_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            optimizer.step()
            ###########################################################

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                    f'grad_norm {norm_meter.val:.3f} ({norm_meter.avg:.3f})\t'
                    f'mem {memory_used:.0f}MB')
