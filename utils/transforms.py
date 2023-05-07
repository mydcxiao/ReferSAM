# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torchvision.transforms import InterpolationMode
from PIL import Image

from copy import deepcopy
from typing import Tuple, List


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: np.ndarray):
        for t in self.transforms:
            image = t(image)
        return image

class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply_image(image)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size)) if len(image.shape) == 3 \
               else np.array(resize(to_pil_image(image), target_size, interpolation=InterpolationMode.NEAREST))
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww) # input_size

class Normalize:
    def __init__(self, 
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375]):
        self.pixel_mean = np.array(pixel_mean)
        self.pixel_std = np.array(pixel_std)
    
    def __call__(self, image):
        return self.preprocess(image)

    def preprocess(self, input_image: np.ndarray) -> torch.Tensor:
        """Normalize pixel values."""
        # Normalize colors
        input_image = (input_image - self.pixel_mean) / self.pixel_std
        return input_image # H x W x C

class ToTensor:
    def __call__(self, image: np.ndarray):
        image = torch.as_tensor(image)
        if len(image.shape) == 3:
            # image = image.permute(2, 0, 1).contiguous()[None, :, :, :]
            image = image.permute(2, 0, 1).contiguous()
        else:
            # image = image.unsqueeze(0).contiguous()[None, :, :, :]
            # image = image.unsqueeze(0).contiguous()
            image = image.contiguous()
        return image # b x c x h x w
    

class Pad:
    def __init__(self, target_length: int):
        self.target_length = target_length
    def __call__(self, image: torch.Tensor):
        """Pad to a square input."""
        # Pad
        h, w = image.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        image = F.pad(image, (0, padw, 0, padh))
        return image # B x C x S x S


