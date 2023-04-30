# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

# from .common import LayerNorm2d

import open_clip

from torch.nn import functional as F


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_name: str,
        pretrained: str,
        input_dim: int,
        depth: int,
        image_embedding_size: Tuple[int, int],
        # activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        # self.act = activation()
        # self.lin = nn.Linear(input_dim, embed_dim)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.text_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.layer = MLP(
            input_dim, input_dim, self.embed_dim, depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_text(x)
        return self.layer(x)
    
    @torch.cuda.amp.autocast()
    def encode_text(self, text: torch.Tensor):
        text = text.squeeze(1)
        x = self.text_model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.text_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_model.transformer(x, attn_mask=self.text_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_model.ln_final(x)
        text_encodings = x
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_model.text_projection
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.unsqueeze(1)
        text_features = torch.cat((text_embed, text_encodings), dim=1)
        return text_features

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

#     def _get_device(self) -> torch.device:
#         return self.point_embeddings[0].weight.device

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

# c = PositionEmbeddingRandom(128)
# for params in c.parameters():
#     print(params.size())
# print(len(list(c.parameters())))