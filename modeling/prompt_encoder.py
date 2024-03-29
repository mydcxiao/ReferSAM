# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d

import open_clip

from torch.nn import functional as F

from timm.models.layers import trunc_normal_


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_name: str,
        pretrained: str,
        text_dim: int,
        depth: int,
        image_embedding_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size

        # self.layer = PositionalLinear(text_dim, self.embed_dim, seq_len=77)
        self.layer = PositionalLinear(text_dim, self.embed_dim, seq_len=1)

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        self.text_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self._freeze_text_model()

    def forward(self, 
                text: torch.Tensor,
                masks: Optional[torch.Tensor],
                ) -> torch.Tensor:
        bs = text.size(0)
        text = self._embed_text(text)
        text_embeddings = self.layer(text)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return text_embeddings, dense_embeddings
    
    @torch.cuda.amp.autocast()
    def _embed_text(self, text: torch.Tensor):
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
        # text_features = torch.cat((text_embed, text_encodings), dim=1)
        # text_features = text_encodings
        text_features = text_embed
        return text_features.float()
    
    def _freeze_text_model(self):
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False
    
    def _freeze_mask(self):
        for p in self.mask_downscaling.parameters():
            p.requires_grad = False
        self.no_mask_embed.weight.requires_grad = False
    
    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_text_model()
        self._freeze_mask()
        return self

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    
    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

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


class PositionalLinear(nn.Module):
    def __init__(self, in_features, out_features, seq_len=77, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, out_features))
        self.norm = nn.LayerNorm(out_features)
        trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        x = self.linear(x)
        x = x + self.positional_embedding
        x = self.norm(x)
        return x
