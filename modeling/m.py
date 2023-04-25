# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class M(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    # @property
    # def device(self) -> Any:
    #     return self.pixel_mean.device

    def forward(
        self,
        batched_images, # B x C x H x W
        batched_sents, # B x 1 x D
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        image_embeddings = self.image_encoder(batched_images)

        text_embeddings = self.prompt_encoder(
            batched_sents,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            # image_embeddings=curr_embedding.unsqueeze(0),
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            text_prompt_embeddings=text_embeddings,
            multimask_output=multimask_output,
        )
        # masks = self.postprocess_masks(
        #     low_res_masks,
        #     input_size=image_record["image"].shape[-2:],
        #     original_size=image_record["original_size"],
        # )
        # masks = masks > self.mask_threshold
        # low_res_masks = low_res_masks > self.mask_threshold
        return low_res_masks, iou_predictions
    
    # def postprocess_masks(
    #     self,
    #     masks: torch.Tensor,
    #     input_size: Tuple[int, ...],
    #     original_size: Tuple[int, ...],
    # ) -> torch.Tensor:
    #     """
    #     Remove padding and upscale masks to the original image size.

    #     Arguments:
    #       masks (torch.Tensor): Batched masks from the mask_decoder,
    #         in BxCxHxW format.
    #       input_size (tuple(int, int)): The size of the image input to the
    #         model, in (H, W) format. Used to remove padding.
    #       original_size (tuple(int, int)): The original size of the image
    #         before resizing for input to the model, in (H, W) format.

    #     Returns:
    #       (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
    #         is given by original_size.
    #     """
    #     masks = F.interpolate(
    #         masks,
    #         (self.image_encoder.img_size, self.image_encoder.img_size),
    #         mode="bilinear",
    #         align_corners=False,
    #     )
    #     masks = masks[..., : input_size[0], : input_size[1]]
    #     masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    #     return masks