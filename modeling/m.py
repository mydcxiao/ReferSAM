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


class m(nn.Module):
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

    # @torch.no_grad()
    def forward(
        self,
        batched_images, # B x C x H x W
        batched_sents, # B x D
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        image_embeddings = self.image_encoder(batched_images)

        # outputs = []
        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
        text_embeddings = self.prompt_encoder(
            # image_record.get("text_feature", None),
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
        # outputs.append(
        #     {
        #         "masks": masks,
        #         "iou_predictions": iou_predictions,
        #         "low_res_logits": low_res_masks,
        #     }
        # )
        # return outputs
        return low_res_masks, iou_predictions
