import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs the model."
        )
    )
    parser.add_argument('--amsgrad', action='store_true',
                    help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--ck_image_encoder', default='./pretrained/image_encoder/sam_vit_h.pth', help='path to checkpoint')
    parser.add_argument('--ck_prompt_encoder', default=None, help='path to checkpoint')
    parser.add_argument('--ck_mask_decoder', default='./pretrained/mask_decoder/sam_vit_h_decoder.pth', help='path to checkpoint')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--img_size', default=1024, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.0008, type=float, help='the initial learning rate')
    parser.add_argument('--layer_ld', default=0.8, type=float, help='the layer-wise learning rate decay')
    parser.add_argument('--model', default='default', help='model: vit_h, vit_l, vit_b')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--warmup', default=50, type=int, help='warmup epochs')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--weight_dice_loss', default=1.0, type=float, help='loss weight for dice loss')
    parser.add_argument('--weight_focal_loss', default=20.0, type=float, help='loss weight for focal loss')
    parser.add_argument('--weight_iou_loss', default=1.0, type=float, help='loss weight for iou prediction')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser