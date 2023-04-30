import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
# from bert.modeling_bert import BertModel

import torchvision

from utils import transforms as T
from utils import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

from criterion import Criterion
from engine import train_one_epoch, eval_batch
from build_m import m_model_registry


def get_dataset(image_set, image_transforms, target_transforms, args):
    from res_dataloader import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=image_transforms,
                      target_transforms=target_transforms,
                      )
    return ds

def get_transform(args):
    image_transforms = [T.ResizeLongestSide(args.img_size),
                        T.Normalize(),
                        T.ToTensor(),
                        T.Pad(args.img_size)
                        ]
    target_transforms = [T.ResizeLongestSide(args.img_size),
                         T.ToTensor(),
                         T.Pad(args.img_size)
                         ]
    
    return T.Compose(image_transforms), T.Compose(target_transforms)


def main(args):
    image_transforms, target_transforms = get_transform(args)

    dataset = get_dataset("train",
                          image_transforms=image_transforms,
                          target_transforms=target_transforms,
                          args=args)
    dataset_test = get_dataset("val",
                               image_transforms=image_transforms,
                               target_transforms=target_transforms,
                               args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = m_model_registry[args.model](resume=None,
                                         ck_image_encoder=args.ck_image_encoder,
                                         ck_prompt_encoder=args.ck_prompt_encoder,
                                         ck_mask_decoder=args.ck_mask_decoder)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    # resume training
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])

    # parameters to optimize
    params = list()
    for name, m in single_model.named_parameters():
        if 'prompt_encoder.layer' in name or 'mask_decoder' in name:
            params.append(m)

    params_to_optimize = [
        {'params': params},
        # {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        # # the following are the parameters of bert
        # {"params": reduce(operator.concat,
        #                     [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
        #                     if p.requires_grad] for i in range(10)])},
    ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999
    
    # criterion for training 
    criterion = Criterion(args.batch_size, 
                          args.weight_focal_loss, 
                          args.weight_dice_loss,
                          args.weight_iou_loss)
    # TODO training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                    iterations)
        iou, overallIoU = eval_batch(model, data_loader_test)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))

            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 
                            'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join('./checkpoints/',
                                                            '{}_best_{}.pth'.format(args.model, args.dataset)))

            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
