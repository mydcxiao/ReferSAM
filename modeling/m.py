# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Optional

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class M(nn.Module):
    # mask_threshold: float = 0.0
    mask_threshold: float = 1.0e-9
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
        batched_masks: Optional[torch.Tensor],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        image_embeddings = self.image_encoder(batched_images)

        text_embeddings, dense_embeddings = self.prompt_encoder(
            batched_sents,
            batched_masks,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            # image_embeddings=curr_embedding.unsqueeze(0),
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            text_prompt_embeddings=text_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        return low_res_masks, iou_predictions