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

# from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

from utils import utils
from utils.transforms import ResizeLongestSide

import open_clip

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        # self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        # self.attention_masks = []
        # self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
        # self.tokenizer = open_clip.get_tokenizer(args.tokenizer)
        self.text_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
        # self.text_encoder, _, _ = open_clip.create_model_and_transforms(args.tokenizer, pretrained=args.text_pretrained)

        # self.eval_mode = eval_mode

        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = self.tokenizer(el['raw'])
                # attention_mask = [0] * self.max_tokens
                # padded_input_ids = [0] * self.max_tokens

                # input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    input_ids = self.text_encoder.encode_text(sentence_raw)
                    input_ids /= input_ids.norm(dim=-1, keepdim=True)

                # truncation of tokens
                # input_ids = input_ids[:self.max_tokens]

                # padded_input_ids[:len(input_ids)] = input_ids
                # attention_mask[:len(input_ids)] = [1]*len(input_ids)

                # sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                sentences_for_ref.append(input_ids)
                # attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            # self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        # if self.eval_mode:
        embedding = []
        # att = []
        for s in range(len(self.input_ids[index])):
            e = self.input_ids[index][s]
            # a = self.attention_masks[index][s]
            embedding.append(e.unsqueeze(-1))
            # att.append(a.unsqueeze(-1))

        tensor_embeddings = torch.cat(embedding, dim=-1)
        # attention_mask = torch.cat(att, dim=-1)
        # else:
        #     choice_sent = np.random.choice(len(self.input_ids[index]))
        #     tensor_embeddings = self.input_ids[index][choice_sent]
        #     attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings#, attention_mask

import torch.distributed as dist
import torch.backends.cudnn as cudnn

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def get_transform(args):
    transforms = [ResizeLongestSide(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
else:
    rank = -1
    world_size = -1

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.distributed.barrier()
setup_for_distributed(is_main_process())

ds = ReferDataset(args,
                  split='val',
                  image_transforms=None,
                  target_transforms=None
                 )

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)  

next(iter(data_loader)) 



