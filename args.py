import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image or directory of images, "
            "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
            "as well as pycocotools if saving in RLE format."
        )
    )

    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     required=True,
    #     help="Path to either a single input image or folder of images.",
    # )

    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     required=True,
    #     help=(
    #         "Path to the directory where masks will be output. Output will be either a folder "
    #         "of PNGs per image or a single json with COCO-style masks."
    #     ),
    # )

    # parser.add_argument(
    #     "--model-type",
    #     type=str,
    #     required=True,
    #     help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    # )

    # parser.add_argument(
    #     "--checkpoint",
    #     type=str,
    #     required=True,
    #     help="The path to the SAM checkpoint to use for mask generation.",
    # )

    # parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    # parser.add_argument(
    #     "--convert-to-rle",
    #     action="store_true",
    #     help=(
    #         "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
    #         "Requires pycocotools."
    #     ),
    # )
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
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser