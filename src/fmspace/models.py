"""图像编码器。

MODELS[name] = (loader, checkpoint)
loader(checkpoint, device) -> (preprocess, forward)
preprocess(images: list[PIL.Image]) -> Tensor[B, C, H, W]，模型的输入（已缩放、标准化）。
forward(x: Tensor[B, C, H, W]) -> Tensor[B, D]，模型的原始输出。

新增模型：写一个 loader，在 MODELS 加一行；loader 需要额外参数时用 functools.partial。
"""

import numpy as np
import torch


def _siglip(checkpoint, device):
    from transformers import AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = SiglipVisionModel.from_pretrained(checkpoint).to(device).eval()

    def preprocess(images):
        return processor(images=images, return_tensors="pt")["pixel_values"].to(device)

    def forward(x):
        return model(pixel_values=x).pooler_output

    return preprocess, forward


def _conceptclip(checkpoint, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()

    def preprocess(images):
        return processor(images=images, return_tensors="pt")["pixel_values"].to(device)

    def forward(x):
        return model.encode_image(image=x)[0]  # (CLS, patch tokens)

    return preprocess, forward


def _flair(checkpoint, device):
    from flair import FLAIRModel

    model = FLAIRModel.from_pretrained(checkpoint).to(device).eval()

    def preprocess(images):
        return torch.cat([model.preprocess_image(np.asarray(im)) for im in images]).to(device)

    def forward(x):
        return model.vision_model(x)

    return preprocess, forward


MODELS = {
    "siglip": (_siglip, "google/siglip-so400m-patch14-384"),
    "medsiglip": (_siglip, "google/medsiglip-448"),
    "conceptclip": (_conceptclip, "JerrryNie/ConceptCLIP"),
    "flair": (_flair, "jusiro2/FLAIR"),
}


def load_model(name, device="cpu"):
    """返回 (preprocess, forward)。"""
    loader, checkpoint = MODELS[name]
    return loader(checkpoint, device)


def load_encoder(name, device="cpu"):
    """返回 encode(images: list[PIL.Image]) -> Tensor[B, D]。"""
    preprocess, forward = load_model(name, device)
    return lambda images: forward(preprocess(images))
