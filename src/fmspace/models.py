"""图像编码器：模型表 + 加载函数。

加载函数：loader(checkpoint, device, **options) -> encode
    encode(images: list[PIL.Image]) -> Tensor[B, D]
    返回模型的原始输出，不归一化。

新增模型：写一个加载函数（或复用已有的），在 MODELS 里加一行。
依赖在加载函数内部 import，没装的包只影响用到它的模型。
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch


def _siglip(checkpoint, device):
    """SigLIP 系列：注意力池化头的输出 pooler_output。"""
    from transformers import AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = SiglipVisionModel.from_pretrained(checkpoint).to(device).eval()

    def encode(images):
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        return model(pixel_values=x).pooler_output

    return encode


def _conceptclip(checkpoint, device):
    """ConceptCLIP：encode_image 的 CLS 向量，只传图像（官方线性探针的写法）。"""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    model = model.to(device).eval()

    def encode(images):
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        out = model.encode_image(image=x)
        return out[0] if isinstance(out, tuple) else out  # (CLS, patch tokens)

    return encode


def _flair(checkpoint, device):
    """FLAIR：vision_model 的输出（含投影层）。"""
    from flair import FLAIRModel

    model = FLAIRModel.from_pretrained(checkpoint).to(device).eval()

    def encode(images):
        x = torch.cat([model.preprocess_image(np.asarray(im)) for im in images])
        return model.vision_model(x.to(device))

    return encode


@dataclass(frozen=True)
class ModelSpec:
    loader: Callable
    checkpoint: str
    options: dict = field(default_factory=dict)  # 原样传给 loader


MODELS = {
    "siglip": ModelSpec(_siglip, "google/siglip-so400m-patch14-384"),
    "medsiglip": ModelSpec(_siglip, "google/medsiglip-448"),
    "conceptclip": ModelSpec(_conceptclip, "JerrryNie/ConceptCLIP"),
    "flair": ModelSpec(_flair, "jusiro2/FLAIR"),
}


def load_encoder(name, device=None):
    """device 为 None 时，有 GPU 用 GPU。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = MODELS[name]
    return spec.loader(spec.checkpoint, torch.device(device), **spec.options)
