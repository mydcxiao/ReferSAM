import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs the model."
        )
    )
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--weight_iou_loss', default=1.0, type=float, help='loss weight for iou prediction')
    parser.add_argument('--weight_focal_loss', default=20.0, type=float, help='loss weight for focal loss')
    parser.add_argument('--weight_dice_loss', default=1.0, type=float, help='loss weight for dice loss')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser