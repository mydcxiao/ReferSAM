import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

import torchvision
from torchvision.transforms.functional import resize, to_pil_image 

from utils import transforms as T
from utils import utils
import numpy as np
import torch.nn.functional as F
import gc
import torch.distributed as dist
from PIL import Image


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
    # TODO check if below code works for freezing in DDP
    #----------------------------------------------------------------
    # model.module.image_encoder.eval()
    # model.module.image_encoder.requires_grad_(False)
    # model.module.prompt_encoder.text_model.eval()
    # model.module.prompt_encoder.text_model.requires_grad_(False)
    #----------------------------------------------------------------
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

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

        low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)

        iou_gt, _, _ = computeIoU(low_res_masks, target) #TODO check if this is correct

        _, idx = iou_gt.max(1)

        max_logits = low_res_logits.clone().detach()[range(len(idx)), idx].unsqueeze(1)

        world_size = utils.get_world_size()
        global_index_list = [torch.zeros_like(index, dtype=index.dtype) for _ in range(world_size)]
        global_pred_list = [torch.zeros_like(max_logits, dtype=max_logits.dtype) for _ in range(world_size)]

        dist.barrier()
        dist.all_gather(global_index_list, index)
        dist.all_gather(global_pred_list, max_logits)

        global_index = torch.cat(global_index_list, dim=0)
        global_pred = torch.cat(global_pred_list, dim=0)

        data_loader.sampler.dataset.set_batch_curr_pred(global_index.cpu(), global_pred.cpu())


        loss = criterion(low_res_logits, target, iou_pred, iou_gt) #TODO check if this is correct

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        #--------------------------------
        # check gradient
        # if epoch == 1:
        #     for n, p in model.module.named_parameters():
        #         if p.grad is not None:
        #             print(n)
                # print(n,':', p.grad)
        #--------------------------------
        optimizer.step()
        if epoch == 0:
            lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        #-----------------------------------------------------------------------------------------------
        # for summary writer
        if writer is not None:
            writer.add_scalar(f'train_per_iter/{utils.get_rank():d}_loss', loss.item(), 
                               iterations + len(data_loader) * epoch)
        #-----------------------------------------------------------------------------------------------

        del image, target, sentences, last_mask, index, data, \
            low_res_logits, iou_pred, low_res_masks, iou_gt, idx, \
            max_logits, global_index_list, global_pred_list, global_index, global_pred, loss

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    metric_logger.synchronize_between_processes()
    train_loss = torch.tensor(train_loss, dtype=torch.float64, device='cuda')
    dist.barrier()
    dist.all_reduce(train_loss)
    train_loss = train_loss.item()
    print("Train loss: {:4f} ({:4f})\n".format(metric_logger.meters['loss'].total, train_loss))
    #-----------------------------------------------------------------------------------------------
    # for summary writer
    if writer is not None:
        writer.add_scalar('train_per_epoch/loss', train_loss, epoch)
    #-----------------------------------------------------------------------------------------------



def eval_train(model, data_loader, epoch, multimask,
               writer = None,
               ):
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
        for data in metric_logger.log_every(data_loader, 50, header):
            total_its += 1
            image, target, sentences, _, _, _, original_img = data

            image, target, sentences = image.cuda(non_blocking=True),\
                                       target.cuda(non_blocking=True),\
                                       sentences.cuda(non_blocking=True),\

            low_res_logits, iou_pred = model(image, sentences, None, multimask_output=multimask)

            low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)

            iou, I, U = computeIoU(low_res_masks, target) # B x C

            iou, idx = iou.max(1)
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

            #-----------------------------------------------------------------------------------------------
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
            #-----------------------------------------------------------------------------------------------

            torch.cuda.synchronize()

            del image, target, sentences, data, \
                low_res_logits, iou_pred, low_res_masks, iou, I, U, idx #, \
                # max_iou_pred, max_iou_pred_gt
            
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)

    t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total], dtype=torch.float64, device='cuda')
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
    seg_correct = seg.tolist()

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    #------------------------------------------------------------------------------------------------
    # output validation iou in summary writer
    if writer is not None:
        writer.add_scalar('eval_val/global_mIoU', mIoU * 100., epoch)
        writer.add_scalar('eval_val/global_average_IoU', 100 * iou, epoch)
        writer.add_scalar('eval_val/global_overall_IoU', cum_I * 100. / cum_U, epoch)
    #------------------------------------------------------------------------------------------------

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
            image, target, sentences, _, original_size, input_size, original_img = data
            image, target, sentences = image.to(device), target.to(device), \
                                       sentences.to(device)
            
            target = target.cpu().data.numpy()
            iterations += 1

            for j in range(sentences.size(-1)):
                low_res_logits, iou_pred = model(image, sentences[:, :, :, j], None, multimask_output=multimask)
                low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)
                max_iou_pred, idx = iou_pred.max(1)
                low_res_masks = low_res_masks[range(len(idx)), idx, :, :]
                low_res_masks = low_res_masks.cpu().data.numpy()
                max_iou_pred = max_iou_pred.item()

                I, U = IoU(low_res_masks, target)
                this_iou = 0.0 if U == 0 else I*1.0/U
                mean_IoU.append(this_iou)
                #----------------------------------------------------
                # output images in summary writer
                if writer is not None:
                    img_ndarray = original_img.permute(0,2,3,1).numpy().astype(np.uint8)
                    target1 = target.astype(np.uint8)
                    target2 = low_res_masks
                    target2 = target2.astype(np.uint8)
                    for i in range(img_ndarray.shape[0]):
                        img = img_ndarray[i, :, : ,:]
                        mask1 = target1[i]
                        mask2 = target2[i]
                        i_size = input_size[i].tolist()
                        o_size = tuple(original_size[i].tolist())
                        visualization1 = overlay_davis(img, mask1, colors=[[0,0,0],[0,255,0]])
                        visualization2 = overlay_davis(img, mask2)
                        visualization = 0.5 * visualization1 + 0.5 * visualization2
                        visualization = visualization.astype(img.dtype)
                        visualization = visualization[ : i_size[0], : i_size[1], :]
                        visualization = np.array(resize(to_pil_image(visualization), o_size))
                        writer.add_image(f'eval_test/{iterations:d}_{this_iou:.2f}_{max_iou_pred:.2f}', visualization, iterations,
                                        dataformats='HWC')
                #----------------------------------------------------
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

                pred_iou += max_iou_pred
        
            del image, target, sentences, data, \
                original_size, input_size, original_img, \
                low_res_logits, iou_pred, low_res_masks, max_iou_pred, idx, I, U, this_iou

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