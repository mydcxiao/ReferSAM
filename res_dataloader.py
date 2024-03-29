import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
from torch import nn

# import h5py
from refer.refer import REFER

from args import get_parser

from utils import utils
from utils import transforms as T

from collections import defaultdict


import open_clip
# from bert.tokenization_bert import BertTokenizer

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False,
                 ):

        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.tokenizer = open_clip.tokenize

        self.eval_mode = eval_mode

        self.last_pred = None
        self.curr_pred = dict()

        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                input_ids = self.tokenizer(el['raw']) # 1 x 77
                sentences_for_ref.append(input_ids)

            self.input_ids.append(sentences_for_ref)

    def set_curr_pred(self, index, mask):
        self.curr_pred[index] = mask
    
    def set_batch_curr_pred(self, indices, masks):
        for i, index in enumerate(indices.tolist()):
            self.set_curr_pred(index, masks[i])

    def update_last_pred(self):
        self.last_pred = self.curr_pred
        self.curr_pred = dict()
    
    def resume(self, pred):
        self.last_pred = pred

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        this_sent = self.refer.Refs[this_ref_id]['sentences']

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        img = np.array(img)

        #--------------------------------------------------------------
        # for visualization
        img_trans = T.Compose([T.ResizeLongestSide(args.img_size // 4),
                               T.ToTensor(),
                               T.Pad(args.img_size // 4)
                              ])
        original_img = img_trans(img)
        #--------------------------------------------------------------

        original_size = torch.Tensor(img.shape[:2])
        input_size = torch.Tensor(T.ResizeLongestSide.get_preprocess_shape(img.shape[0], 
                                                                           img.shape[1], 
                                                                           args.img_size // 4))

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        target = np.array(annot)

        if self.image_transforms is not None:
            img = self.image_transforms(img)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        
        if self.eval_mode:
            embedding = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                embedding.append(e.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            # tensor_embeddings = self.input_ids[index][choice_sent].squeeze(0)
        
        if self.eval_mode:
            return img, target.float(), tensor_embeddings, original_size, input_size, original_img, this_sent

        if self.last_pred:
            if index in self.last_pred:
                last_mask = self.last_pred[index]
            else:
                last_mask = torch.zeros_like(list(self.last_pred.values())[0])
            
            return img, target.float(), tensor_embeddings, last_mask, index, original_size, input_size, original_img

        return img, target.float(), tensor_embeddings, index, original_size, input_size, original_img
        


# import torch.distributed as dist
# import torch.backends.cudnn as cudnn

# def get_transform(args):
#     from utils import transforms as T
#     image_transforms = [T.ResizeLongestSide(args.img_size),
#                          T.Normalize(),
#                          T.ToTensor(),
#                          T.Pad(args.img_size)
#                         ]
#     target_transforms = [T.ResizeLongestSide(args.img_size),
#                          T.ToTensor(),
#                          T.Pad(args.img_size)
#                         ]
    
#     return T.Compose(image_transforms), T.Compose(target_transforms)

# from utils import utils
# utils.init_distributed_mode(args)

# image_transforms, target_transforms = get_transform(args)

# ds = ReferDataset(args,
#                   split='val',
#                   image_transforms=image_transforms,
#                   target_transforms=target_transforms,
#                 #   eval_mode=False,
#                 #   eval_mode = True,
#                  )

# num_tasks = utils.get_world_size()
# global_rank = utils.get_rank()
# train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=num_tasks, rank=global_rank,
#                                                                     shuffle=True)
# data_loader = torch.utils.data.DataLoader(
#         ds, batch_size=args.batch_size,
#         sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)  


# test_sampler = torch.utils.data.SequentialSampler(ds)
# data_loader = torch.utils.data.DataLoader(ds, batch_size=1,
#                                                 sampler=test_sampler, num_workers=args.workers)


# print(next(iter(data_loader))[0].size()) 
# print(next(iter(data_loader))[1].size()) 
# print(next(iter(data_loader))[2].size()) 
# print(next(iter(data_loader))[3].size()) 
# print(next(iter(data_loader))[4].size())
# print(next(iter(data_loader))[5].size())
# print(next(iter(data_loader))[6].size())

# print(next(iter(data_loader))[0].dtype)
# print(next(iter(data_loader))[1].dtype) 
# print(next(iter(data_loader))[2].dtype) 
# print(next(iter(data_loader))[3].dtype) 
# print(next(iter(data_loader))[4].dtype)
# print(next(iter(data_loader))[5].dtype)
# print(next(iter(data_loader))[6].dtype)


# print(next(iter(data_loader))[1])
# print(next(iter(data_loader))[3]) 
# print(next(iter(data_loader))[4])


