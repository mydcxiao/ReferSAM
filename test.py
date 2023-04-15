import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

# from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
# import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# import argparse
import json
# import os
from typing import Any, Dict, List


def get_dataset(image_set, transform, args):
    from res_dataloader import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    # num_classes = 2
    return ds#, num_classes


# def evaluate(model, data_loader, bert_model, device):
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")

#     # evaluation variables
#     cum_I, cum_U = 0, 0
#     eval_seg_iou_list = [.5, .6, .7, .8, .9]
#     seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
#     seg_total = 0
#     mean_IoU = []
#     header = 'Test:'

#     with torch.no_grad():
#         for data in metric_logger.log_every(data_loader, 100, header):
#             image, target, sentences, attentions = data
#             image, target, sentences, attentions = image.to(device), target.to(device), \
#                                                    sentences.to(device), attentions.to(device)
#             sentences = sentences.squeeze(1)
#             attentions = attentions.squeeze(1)
#             target = target.cpu().data.numpy()
#             for j in range(sentences.size(-1)):
#                 if bert_model is not None:
#                     last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
#                     embedding = last_hidden_states.permute(0, 2, 1)
#                     output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
#                 else:
#                     output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

#                 output = output.cpu()
#                 output_mask = output.argmax(1).data.numpy()
#                 I, U = computeIoU(output_mask, target)
#                 if U == 0:
#                     this_iou = 0.0
#                 else:
#                     this_iou = I*1.0/U
#                 mean_IoU.append(this_iou)
#                 cum_I += I
#                 cum_U += U
#                 for n_eval_iou in range(len(eval_seg_iou_list)):
#                     eval_seg_iou = eval_seg_iou_list[n_eval_iou]
#                     seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
#                 seg_total += 1

#             del image, target, sentences, attentions, output, output_mask
#             if bert_model is not None:
#                 del last_hidden_states, embedding

#     mean_IoU = np.array(mean_IoU)
#     mIoU = np.mean(mean_IoU)
#     print('Final results:')
#     print('Mean IoU is %.2f\n' % (mIoU*100.))
#     results_str = ''
#     for n_eval_iou in range(len(eval_seg_iou_list)):
#         results_str += '    precision@%s = %.2f\n' % \
#                        (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
#     results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
#     print(results_str)


# def get_transform(args):
#     transforms = [T.Resize(args.img_size, args.img_size),
#                   T.ToTensor(),
#                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                   ]

#     return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

# def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
#     header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
#     metadata = [header]
#     for i, mask_data in enumerate(masks):
#         mask = mask_data["segmentation"]
#         filename = f"{i}.png"
#         cv2.imwrite(os.path.join(path, filename), mask * 255)
#         mask_metadata = [
#             str(i),
#             str(mask_data["area"]),
#             *[str(x) for x in mask_data["bbox"]],
#             *[str(x) for x in mask_data["point_coords"][0]],
#             str(mask_data["predicted_iou"]),
#             str(mask_data["stability_score"]),
#             *[str(x) for x in mask_data["crop_box"]],
#         ]
#         row = ",".join(mask_metadata)
#         metadata.append(row)
#     metadata_path = os.path.join(path, "metadata.csv")
#     with open(metadata_path, "w") as f:
#         f.write("\n".join(metadata))

#     return


def main(args):
    # device = torch.device(args.device)
    # dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
    #                                                sampler=test_sampler, num_workers=args.workers)
    # print(args.model)
    # single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # single_model.load_state_dict(checkpoint['model'])
    # model = single_model.to(device)

    # if args.model != 'lavt_one':
    #     model_class = BertModel
    #     single_bert_model = model_class.from_pretrained(args.ck_bert)
    #     # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
    #     if args.ddp_trained_weights:
    #         single_bert_model.pooler = None
    #     single_bert_model.load_state_dict(checkpoint['bert_model'])
    #     bert_model = single_bert_model.to(device)
    # else:
    #     bert_model = None

    # evaluate(model, data_loader_test, bert_model, device=device)


    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    print(args.model_type)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    # amg_kwargs = get_amg_kwargs(args)
    # generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    
    # if not os.path.isdir(args.input):
    #     targets = [args.input]
    # else:
    #     targets = [
    #         f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
    #     ]
    #     targets = [os.path.join(args.input, f) for f in targets]

    # os.makedirs(args.output, exist_ok=True)

    # for t in targets:
    #     print(f"Processing '{t}'...")
    #     image = cv2.imread(t)
    #     if image is None:
    #         print(f"Could not load '{t}' as an image, skipping...")
    #         continue
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     masks = generator.generate(image)

    #     base = os.path.basename(t)
    #     base = os.path.splitext(base)[0]
    #     save_base = os.path.join(args.output, base)
    #     if output_mode == "binary_mask":
    #         os.makedirs(save_base, exist_ok=False)
    #         write_masks_to_folder(masks, save_base)
    #     else:
    #         save_file = save_base + ".json"
    #         with open(save_file, "w") as f:
    #             json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # print('Image size: {}'.format(str(args.img_size)))
    main(args)
