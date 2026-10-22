"""图像编码器。

MODELS[name] = (loader, checkpoint)
loader(checkpoint, device) -> encode
encode(images: list[PIL.Image]) -> Tensor[B, D]，模型的原始输出。

新增模型：写一个 loader，在 MODELS 加一行；loader 需要额外参数时用 functools.partial。
"""

import numpy as np
import torch


def _siglip(checkpoint, device):
    from transformers import AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = SiglipVisionModel.from_pretrained(checkpoint).to(device).eval()

    def encode(images):
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        return model(pixel_values=x).pooler_output

    return encode


def _conceptclip(checkpoint, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()

    def encode(images):
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        return model.encode_image(image=x)[0]  # (CLS, patch tokens)

    return encode


def _flair(checkpoint, device):
    from flair import FLAIRModel

    model = FLAIRModel.from_pretrained(checkpoint).to(device).eval()

    def encode(images):
        x = torch.cat([model.preprocess_image(np.asarray(im)) for im in images])
        return model.vision_model(x.to(device))

    return encode


MODELS = {
    "siglip": (_siglip, "google/siglip-so400m-patch14-384"),
    "medsiglip": (_siglip, "google/medsiglip-448"),
    "conceptclip": (_conceptclip, "JerrryNie/ConceptCLIP"),
    "flair": (_flair, "jusiro2/FLAIR"),
}


def load_encoder(name, device="cpu"):
    loader, checkpoint = MODELS[name]
    return loader(checkpoint, device)
