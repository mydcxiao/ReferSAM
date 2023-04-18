import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image or directory of images, "
            "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
            "as well as pycocotools if saving in RLE format."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    # amg_settings = parser.add_argument_group("AMG Settings")

    # amg_settings.add_argument(
    #     "--points-per-side",
    #     type=int,
    #     default=None,
    #     help="Generate masks by sampling a grid over the image with this many points to a side.",
    # )

    # amg_settings.add_argument(
    #     "--points-per-batch",
    #     type=int,
    #     default=None,
    #     help="How many input points to process simultaneously in one batch.",
    # )

    # amg_settings.add_argument(
    #     "--pred-iou-thresh",
    #     type=float,
    #     default=None,
    #     help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    # )

    # amg_settings.add_argument(
    #     "--stability-score-thresh",
    #     type=float,
    #     default=None,
    #     help="Exclude masks with a stability score lower than this threshold.",
    # )

    # amg_settings.add_argument(
    #     "--stability-score-offset",
    #     type=float,
    #     default=None,
    #     help="Larger values perturb the mask more when measuring stability score.",
    # )

    # amg_settings.add_argument(
    #     "--box-nms-thresh",
    #     type=float,
    #     default=None,
    #     help="The overlap threshold for excluding a duplicate mask.",
    # )

    # amg_settings.add_argument(
    #     "--crop-n-layers",
    #     type=int,
    #     default=None,
    #     help=(
    #         "If >0, mask generation is run on smaller crops of the image to generate more masks. "
    #         "The value sets how many different scales to crop at."
    #     ),
    # )

    # amg_settings.add_argument(
    #     "--crop-nms-thresh",
    #     type=float,
    #     default=None,
    #     help="The overlap threshold for excluding duplicate masks across different crops.",
    # )

    # amg_settings.add_argument(
    #     "--crop-overlap-ratio",
    #     type=int,
    #     default=None,
    #     help="Larger numbers mean image crops will overlap more.",
    # )

    # amg_settings.add_argument(
    #     "--crop-n-points-downscale-factor",
    #     type=int,
    #     default=None,
    #     help="The number of points-per-side in each layer of crop is reduced by this factor.",
    # )

    # amg_settings.add_argument(
    #     "--min-mask-region-area",
    #     type=int,
    #     default=None,
    #     help=(
    #         "Disconnected mask regions or holes with area smaller than this value "
    #         "in pixels are removed by postprocessing."
    #     ),
    # )
    return parser