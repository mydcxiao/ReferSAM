import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

# from .common import LayerNorm2d

import open_clip

model, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
# text = tokenizer(["a diagram", "a dog", "a cat"])
tokenizer = open_clip.tokenize
text = tokenizer(["a diagram", 'a dog', 'a cat'])
# text = open_clip.tokenize("a diagram")
# text = tokenizer(["a diagram"])
# text = tokenizer("photo of a diagram")
# print(text)
val, idx = text.max(dim=-1)
print(text.size())
print(val)
print(idx)
print(model.text_projection.size())
text_mask = (text != 0).long()
x = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
x = x + model.positional_embedding
x = x.permute(1, 0, 2)  # NLD -> LND
x = model.transformer(x, attn_mask=model.attn_mask)
# x = model.transformer(x, attn_mask=text_mask)
x = x.permute(1, 0, 2)  # LND -> NLD
x = model.ln_final(x)
text_encodings = x
text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection
print(text_encodings.size(), text_embed.size())
print(model.attn_mask.size())


# text = text.unsqueeze(1)
# print(text.size())
# text = text[:, :20]
# for param in model.parameters():
#     # param.requires_grad = False
#     print(param.requires_grad)
# with torch.no_grad(), torch.cuda.amp.autocast():
# #     # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
# #     # image_features /= image_features.norm(dim=-1, keepdim=True)
# #     text_features /= text_features.norm(dim=-1, keepdim=True)
#     # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# print(text_features)
# print(text_features.size()) # 3 * 768
# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]