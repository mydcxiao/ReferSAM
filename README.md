# ReferSAM
The official PyTorch implementation of SAM model for refering image segmentation(RIS).

## Framework
More details([SAM](https://github.com/facebookresearch/segment-anything))
<p align="center">
  <img src="./referSAM.png" width="1000">
</p>

## Preparation

1. Environment
   - refer to [SAM](https://github.com/facebookresearch/segment-anything)
2. Datasets
   - The detailed instruction is in [LAVT](https://github.com/yz93/LAVT-RIS).
3. Pretrained weights
   - refer to [SAM](https://github.com/facebookresearch/segment-anything)

## Train and Test
Training with 3 V-100s GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node 3 train.py --model vit_h --dataset refcoco --split train --batch-size 8 --epochs 40 --img_size 1024 --lr 0.0001 2>&1 | tee ./logs/refcoco/vit_h_output
```
Testing
```shell
python test.py --model vit_h --dataset refcoco --split testB --resume ./checkpoints/vit_h_best_refcoco.pth --img_size 1024 --multimask
```
Babysitting
```shell
tensorboard --logdir ./logs/vit_h_refcoco_test/ --port 6006
```
More details, refer to [LAVT](https://github.com/yz93/LAVT-RIS).

## Results
|     Dataset     | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Overall IoU | Mean IoU |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:--------:|
| RefCOCO val     | 79.50 | 74.00 | 67.45 | 55.47 | 22.93 |    64.64    |   71.06  |
| RefCOCO test A  | 83.03 | 78.20 | 71.68 | 58.60 | 22.38 |    68.61    |   73.35  |
| RefCOCO test B  | 73.68 | 67.11 | 60.22 | 49.44 | 26.79 |    59.96    |   67.79  |

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.


Some code changes come from [SAM](https://github.com/facebookresearch/segment-anything) and [LAVT](https://github.com/yz93/LAVT-RIS).
