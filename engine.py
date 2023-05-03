import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

import torchvision

from utils import transforms as T
from utils import utils
import numpy as np
import torch.nn.functional as F
import gc
import torch.distributed as dist

def computeIoU(pred, # B x C x H x W 
               gt, # B x H x W
               ):
    gt = gt.unsqueeze(1)
    gt = gt.repeat_interleave(pred.size(1),dim=1)
    I = torch.sum(torch.mul(pred, gt), (3, 2)) # B x C
    U = torch.sum(torch.add(pred, gt), (3, 2)) - I # B x C
    iou = torch.full_like(U, fill_value=0.0)
    mask = (U != 0)
    iou[mask] = I[mask] / U[mask]
    return iou, I, U # B x C

def IoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations):
    model.train()
    model.module.image_encoder.eval()
    model.module.image_encoder.requires_grad_(False)
    model.module.prompt_encoder.text_model.eval()
    model.module.prompt_encoder.text_model.requires_grad_(False)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, original_size, input_size = data
        image, target, sentences, original_size, input_size = image.cuda(non_blocking=True),\
                                                              target.cuda(non_blocking=True),\
                                                              sentences.cuda(non_blocking=True),\
                                                              original_size.cuda(non_blocking=True),\
                                                              input_size.cuda(non_blocking=True)
                                                            #   attentions.cuda(non_blocking=True)

        mask, iou_pred = model(image, sentences, multimask_output=True)

        iou_gt, _, _ = computeIoU(mask, target) #TODO check

        loss = criterion(mask.float(), target, iou_pred, iou_gt) #TODO check

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, original_size, input_size, loss, mask, iou_pred, data

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    metric_logger.synchronize_between_processes()
    train_loss = torch.tensor(train_loss, dtype=torch.float64, device='cuda')
    dist.barrier()
    dist.all_reduce(train_loss)
    train_loss = train_loss.item()
    print("Train loss: {:4f} ({:4f})".format(metric_logger.meters['loss'].total, train_loss))


def eval_batch(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, original_size, input_size = data
            image, target, sentences, original_size, input_size = image.cuda(non_blocking=True),\
                                                                  target.cuda(non_blocking=True),\
                                                                  sentences.cuda(non_blocking=True),\
                                                                  original_size.cuda(non_blocking=True),\
                                                                  input_size.cuda(non_blocking=True)
                                                                #   attentions.cuda(non_blocking=True)

            masks, iou_pred = model(image, sentences, multimask_output=True)

            iou, I, U = computeIoU(masks, target) # B x C

            iou, idx = iou.max(1)
            I = I[range(len(idx)), idx]
            U = U[range(len(idx)), idx]

            iou = iou.sum().item() / iou.size(0)
            I = I.sum().item()
            U = U.sum().item()

            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def eval_seq(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    pred_iou = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, original_size, input_size = data
            image, target, sentences, original_size, input_size = image.to(device), target.to(device), \
                                                                  sentences.to(device), original_size.to(device), \
                                                                  input_size.to(device)
                                                                  #, attentions.to(device)
            
            target = target.cpu().data.numpy()

            for j in range(sentences.size(-1)):
                masks, iou_pred = model(image, sentences[:, :, j], multimask_output=True)
                min_iou_pred, idx = iou_pred.min(1)
                output_mask = masks[:, idx.item(), :, :]
                output_mask = output_mask.cpu().data.numpy()

                I, U = IoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

                pred_iou += min_iou_pred

            del image, target, sentences, original_size, input_size, \
                masks, iou_pred, min_iou_pred, idx, output_mask

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))

    print('Predicted IoU is %.2f\n' % (pred_iou*100.))

    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)