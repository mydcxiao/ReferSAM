import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

# from .common import LayerNorm2d

import open_clip

model, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
text = tokenizer(["a diagram", "a dog", "a cat"])
# text = tokenizer(["a diagram"])
# text = tokenizer("photo of a diagram")
print(text)
print(text.size())
# text = text.unsqueeze(1)
# print(text.size())
# text = text[:, :20]
# for param in model.parameters():
#     # param.requires_grad = False
#     print(param.requires_grad)
with torch.no_grad(), torch.cuda.amp.autocast():
# #     # image_features = model.encode_image(image)
    text_features = model.encode_text(text)
# #     # image_features /= image_features.norm(dim=-1, keepdim=True)
# #     text_features /= text_features.norm(dim=-1, keepdim=True)
#     # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# print(text_features)
print(text_features.size()) # 3 * 768
# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]