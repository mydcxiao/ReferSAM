import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

import torchvision
from torchvision.transforms.functional import resize, to_pil_image 
from torchvision.transforms import InterpolationMode

from utils import transforms as T
from utils import utils
import numpy as np
import torch.nn.functional as F
import gc
import torch.distributed as dist
from PIL import Image

from typing import Tuple, List


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


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, 
                    epoch, print_freq, iterations, 
                    multimask,
                    writer = None,
                    ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    total_loss = 0
    total_its = 0

    #evaluation variables
    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []


    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        last_mask = None
        if epoch == 0:
            image, target, sentences, index, _, _, _ = data
            image, target, sentences, index = image.cuda(non_blocking=True),\
                                            target.cuda(non_blocking=True),\
                                            sentences.cuda(non_blocking=True),\
                                            index.cuda(non_blocking=True)
        else:
            image, target, sentences, last_mask, index, _, _, _ = data
            image, target, sentences, last_mask, index = image.cuda(non_blocking=True),\
                                                        target.cuda(non_blocking=True),\
                                                        sentences.cuda(non_blocking=True),\
                                                        last_mask.cuda(non_blocking=True),\
                                                        index.cuda(non_blocking=True)


        low_res_logits, iou_pred = model(image, sentences, last_mask, multimask_output=multimask)

        # low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)
        soft_low_res_masks = torch.sigmoid(low_res_logits)

        iou_gt, _, _ = computeIoU(soft_low_res_masks, target)

        # _, idx = iou_gt.max(1)
        _, idx = iou_pred.max(1)

        max_logits = low_res_logits.clone().detach()[range(len(idx)), idx].unsqueeze(1)

        world_size = utils.get_world_size()
        global_index_list = [torch.zeros_like(index, dtype=index.dtype) for _ in range(world_size)]
        global_pred_list = [torch.zeros_like(max_logits, dtype=max_logits.dtype) for _ in range(world_size)]

        dist.barrier()
        dist.all_gather(global_index_list, index)
        dist.all_gather(global_pred_list, max_logits)

        global_index = torch.cat(global_index_list, dim=0)
        global_pred = torch.cat(global_pred_list, dim=0)

        data_loader.dataset.set_batch_curr_pred(global_index.cpu(), global_pred.cpu())


        loss = criterion(low_res_logits, target, iou_pred, iou_gt) #TODO check if this is correct

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        total_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # for evaluation
        with torch.no_grad():
            low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)
            iou, I, U = computeIoU(low_res_masks, target) # B x C
            iou, idx = iou.max(1)
            I = I[range(len(idx)), idx]
            U = U[range(len(idx)), idx]

            acc_ious += iou.mean().item()
            mean_IoU.append(iou.mean().item())
            cum_I += I.sum().item()
            cum_U += U.sum().item()

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                for i in range(len(idx)):
                    if iou[i].item() >= eval_seg_iou:
                        seg_correct[n_eval_iou] += 1
            seg_total += len(idx)


        # for summary writer
        if writer is not None:
            writer.add_scalar(f'train_per_iter/{utils.get_rank():d}_loss', loss.item(), 
                               iterations + len(data_loader) * epoch)
            writer.add_scalar('train_per_iter/lr', optimizer.param_groups[0]["lr"], 
                               iterations + len(data_loader) * epoch)

        del image, target, sentences, last_mask, index, data, \
            low_res_logits, iou_pred, soft_low_res_masks, iou_gt, idx, max_logits, global_index_list, global_pred_list, global_index, global_pred, loss, \
            low_res_masks, iou, I, U

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # metric_logger.synchronize_between_processes()
    # total_loss = torch.tensor(total_loss, dtype=torch.float64, device='cuda')
    # dist.barrier()
    # dist.all_reduce(total_loss)
    # total_loss = total_loss.item()
    # print("Train loss: {:4f} ({:4f})\n".format(metric_logger.meters['loss'].total, total_loss))
    
    iou = acc_ious / total_its
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total, total_loss], dtype=torch.float64, device='cuda')
    seg = torch.tensor(seg_correct, dtype=torch.int32, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    dist.all_reduce(seg)
    t = t.tolist()
    iou = t[0] / utils.get_world_size()
    mIoU = t[1] / utils.get_world_size()
    cum_I = t[2]
    cum_U = t[3]
    seg_total = int(t[4])
    total_loss = t[5]
    seg_correct = seg.tolist()

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    # for summary writer
    if writer is not None:
        writer.add_scalar('train_per_epoch/loss', total_loss, epoch)
        writer.add_scalar('train_per_epoch/mIoU', mIoU * 100., epoch)
        writer.add_scalar('train_per_epoch/average_IoU', 100 * iou, epoch)
        writer.add_scalar('train_per_epoch/overall_IoU', cum_I * 100. / cum_U, epoch)




def eval_train(model, data_loader, epoch, multimask, criterion,
               writer = None,
               ):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0

    # evaluation variables
    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    total_loss = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 50, header):
            total_its += 1
            image, target, sentences, _, _, _, original_img = data

            image, target, sentences = image.cuda(non_blocking=True),\
                                       target.cuda(non_blocking=True),\
                                       sentences.cuda(non_blocking=True),\

            low_res_logits, iou_pred = model(image, sentences, None, multimask_output=multimask)

            # compute loss for validation set
            soft_low_res_masks = torch.sigmoid(low_res_logits)

            iou_gt, _, _ = computeIoU(soft_low_res_masks, target) 

            loss = criterion(low_res_logits, target, iou_pred, iou_gt) #TODO check if this is correct

            total_loss += loss.item()


            low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)

            iou, I, U = computeIoU(low_res_masks, target) # B x C

            # iou, idx = iou.max(1)
            _, idx = iou_pred.max(1)
            iou = iou[range(len(idx)), idx]
            I = I[range(len(idx)), idx]
            U = U[range(len(idx)), idx]

            # max_iou_pred_gt = iou_pred[range(len(idx)), idx]
            # max_iou_pred, _ = iou_pred.max(1)

            acc_ious += iou.mean().item()
            mean_IoU.append(iou.mean().item())
            cum_I += I.sum().item()
            cum_U += U.sum().item()

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                for i in range(len(idx)):
                    if iou[i].item() >= eval_seg_iou:
                        seg_correct[n_eval_iou] += 1
            seg_total += len(idx)

            # for summary writer
            if writer is not None:
                img_ndarray = original_img.permute(0,2,3,1).numpy().astype(np.uint8)
                target1 = target.cpu().data.numpy().astype(np.uint8)
                target2 = low_res_masks[range(len(idx)), idx].squeeze(1).cpu().data.numpy()
                target2 = target2.astype(np.uint8)
                for i in range(img_ndarray.shape[0]):
                    img = img_ndarray[i, :, : ,:]
                    mask1 = target1[i]
                    mask2 = target2[i]
                    visualization1 = overlay_davis(img, mask1, colors=[[0,0,0],[0,255,0]])
                    visualization2 = overlay_davis(img, mask2)
                    visualization = 0.5 * visualization1 + 0.5 * visualization2
                    visualization = visualization.astype(img.dtype)
                    writer.add_image(f'eval_val/{utils.get_rank():d}_{total_its:d}_{i:d}'
                                     , visualization, epoch, dataformats='HWC')
                    writer.add_text(f'eval_val/{utils.get_rank():d}_{total_its:d}_{i:d}',
                                    f'{loss.item():.4f}', epoch)

            torch.cuda.synchronize()

            del image, target, sentences, data, \
                low_res_logits, iou_pred, low_res_masks, iou, I, U, idx, \
                soft_low_res_masks, iou_gt, loss  # max_iou_pred, max_iou_pred_gt
            
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)

    t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total, total_loss], dtype=torch.float64, device='cuda')
    # t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total], dtype=torch.float64, device='cuda')
    seg = torch.tensor(seg_correct, dtype=torch.int32, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    dist.all_reduce(seg)
    t = t.tolist()
    iou = t[0] / utils.get_world_size()
    mIoU = t[1] / utils.get_world_size()
    cum_I = t[2]
    cum_U = t[3]
    seg_total = int(t[4])
    total_loss = t[5]
    seg_correct = seg.tolist()

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    # output validation iou in summary writer
    if writer is not None:
        writer.add_scalar('eval_val/global_mIoU', mIoU * 100., epoch)
        writer.add_scalar('eval_val/global_average_IoU', 100 * iou, epoch)
        writer.add_scalar('eval_val/global_overall_IoU', cum_I * 100. / cum_U, epoch)
        writer.add_scalar('eval_val/global_loss', total_loss, epoch)

    return 100 * iou, 100 * cum_I / cum_U



def eval_test(model, data_loader, device, multimask, writer=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    iterations = 0
    pred_iou = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, original_size, input_size, original_img, this_sent = data
            image, target, sentences = image.to(device), target.to(device), \
                                       sentences.to(device)
            
            iterations += 1

            for j in range(sentences.size(-1)):
                low_res_logits, iou_pred = model(image, sentences[:, :, :, j], None, multimask_output=multimask)
                low_res_masks = torch.where(low_res_logits > 0.0, 1.0, 0.0)
                iou_gt, _, _ = computeIoU(low_res_masks, target) # B x C
                max_iou_pred, idx = iou_pred.max(1)
                _, idx_gt = iou_gt.max(1)
                idx_choice = idx
                all_masks = low_res_masks.cpu().data.numpy()
                low_res_masks = low_res_masks[range(len(idx_choice)), idx_choice, :, :]
                low_res_masks = low_res_masks.cpu().data.numpy()
                max_iou_pred = max_iou_pred.item()

                target_ = target.cpu().data.numpy()
                I, U = IoU(low_res_masks, target_)
                this_iou = 0.0 if U == 0 else I*1.0/U
                mean_IoU.append(this_iou)
                sent_id = this_sent[j]['sent_id'].item()
                sent = this_sent[j]['sent'][0]
                #----------------------------------------------------
                # output images in summary writer
                if writer is not None:
                    img_ndarray = original_img.permute(0,2,3,1).numpy().astype(np.uint8)
                    target1 = target_.astype(np.uint8)
                    target2 = low_res_masks
                    target2 = target2.astype(np.uint8)
                    all_masks = all_masks.astype(np.uint8)
                    for i in range(img_ndarray.shape[0]):
                        img = img_ndarray[i, :, : ,:]
                        mask1 = target1[i]
                        mask2 = target2[i]
                        i_size = input_size[i].int().tolist()
                        o_size = tuple(original_size[i].int().tolist())
                        visualization1 = overlay_davis(img, mask1, colors=[[0,0,0],[0,255,0]])
                        visualization2 = overlay_davis(img, mask2)
                        visualization = 0.5 * visualization1 + 0.5 * visualization2
                        visualization = visualization.astype(img.dtype)
                        visualization = visualization[ : i_size[0], : i_size[1], :]
                        visualization = np.array(resize(to_pil_image(visualization), o_size))
                        writer.add_image(f'eval_test/{this_iou:.2f}_{idx_choice.item():d}_{idx.item():d}_{idx_gt.item():d}_{sent}', 
                            visualization, dataformats='HWC')
                        vis_list = []
                        colors_list = [[[0,0,0],[255,0,0]], [[0,0,0],[0,255,0]], [[0,0,0],[0,0,255]]]
                        for j in range(all_masks.shape[1]):
                            vis = overlay_davis(img, all_masks[i, j, :, :], colors=colors_list[j])
                            vis = vis[ : i_size[0], : i_size[1], :]
                            vis = np.array(resize(to_pil_image(vis), o_size))
                            vis_list.append(vis)
                        vis_all = vis_list[0] / 3 + vis_list[1] / 3 + vis_list[2] / 3
                        vis_all = vis_all.astype(img.dtype)
                        writer.add_image(f'all_masks/{this_iou:.2f}_{idx_choice.item():d}_{idx.item():d}_{idx_gt.item():d}_{sent}',
                            vis_all, dataformats='HWC')
                #----------------------------------------------------
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

                pred_iou += max_iou_pred
        
            del image, target, sentences, data, \
                original_size, input_size, original_img, target_, \
                low_res_logits, iou_pred, low_res_masks, max_iou_pred, idx, idx_gt, I, U, this_iou

        pred_iou = pred_iou / seg_total

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



# overlay mask and image for visualization
# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def postprocess_masks(
        model,
        masks,
        # input_size: Tuple[int, ...],
        # original_size: Tuple[int, ...],
    ):
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
        """
    # masks = F.interpolate(
    #     masks,
    #     (model.image_encoder.img_size, model.image_encoder.img_size),
    #     mode="nearest",
        # align_corners=False,
    # )
    masks = np.array(resize(to_pil_image(masks), (model.image_encoder.img_size, model.image_encoder.img_size), 
                     interpolation=InterpolationMode.NEAREST))
    # masks = masks[..., : input_size[0], : input_size[1]]
    # masks = F.interpolate(masks, original_size, mode="nearest", align_corners=False)
    return masks