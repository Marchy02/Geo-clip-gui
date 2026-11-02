import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def _as_tensor(features):
    """Normalise the output of ``CLIPModel.get_image_features``.

    Newer ``transformers`` releases may return a ``BaseModelOutputWithPooling``
    (or a tuple) instead of a plain tensor; the MLP head needs the tensor.
    """
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "pooler_output"):
        value = getattr(features, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(features, (tuple, list)):
        return features[0]
    raise TypeError(f"Output CLIP inatteso: {type(features)!r}")


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CLIP = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        self.image_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID)
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]

    def forward(self, x):
        x = _as_tensor(self.CLIP.get_image_features(pixel_values=x))
        return self.mlp(x)
