"""GeoCLIP: CLIP-inspired alignment between images and GPS locations.

This is the neural engine used by the OSINT toolkit.  Compared with the
research code it was forked from, this module:

* never touches hardcoded cluster paths: snapshots are resolved relative to
  the package and weights are loaded with ``map_location`` so CPU-only
  machines work out of the box;
* exposes *presets* that pair a GPS gallery with the matching weights
  (``"world"`` = original MP-16 model, ``"ukraine_russia"`` = fine-tuned
  model + regional gallery);
* accepts a file path, a ``PIL.Image`` or raw bytes in :meth:`predict`.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import file_dir, load_gps_data

log = logging.getLogger(__name__)

ImageInput = str | os.PathLike | Image.Image | bytes

WEIGHTS_DIR = Path(file_dir) / "weights"
GALLERY_DIR = Path(file_dir) / "gps_gallery"
SNAPSHOTS_DIR = Path(file_dir).parent / "snapshots"


@dataclass(frozen=True)
class Preset:
    """A GPS gallery plus the weights it should be used with."""

    gallery: str
    timestamp: str | None
    description: str


PRESETS: dict[str, Preset] = {
    "world": Preset(
        gallery="coordinates_100K.csv",
        timestamp=None,
        description="Modello originale GeoCLIP (MP-16), galleria mondiale da 100K punti",
    ),
    "ukraine_russia": Preset(
        gallery="coordinates_ukraine_russia.csv",
        timestamp="09-20-09:11",
        description="Modello fine-tuned su Ucraina/Russia, galleria regionale",
    ),
}
DEFAULT_PRESET = "world"


def _torch_load(path: Path):
    """Load a tensor/state-dict on CPU, safely when supported."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


def _open_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    else:
        img = Image.open(image)
    return img.convert("RGB")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GeoCLIP(nn.Module):
    def __init__(
        self,
        timestamp: str | None = None,
        epoch_num: int = 0,
        epochs: bool = False,
        from_pretrained: bool = True,
        queue_size: int = 4096,
        preset: str | None = None,
        gallery: str | os.PathLike | None = None,
    ):
        """
        Args:
            timestamp: suffix of a fine-tuned weight set (e.g. ``"09-20-09:11"``).
                Overrides the preset's weights.
            epoch_num / epochs: load a training snapshot from ``geoclip/snapshots``.
            from_pretrained: load weights (disable for training from scratch).
            queue_size: size of the GPS queue used during training.
            preset: ``"world"`` (default) or ``"ukraine_russia"``.
            gallery: explicit GPS gallery CSV (file name inside ``gps_gallery``
                or absolute path). Overrides the preset's gallery.
        """
        super().__init__()
        preset_name = preset or DEFAULT_PRESET
        if preset_name not in PRESETS:
            raise ValueError(f"Preset sconosciuto '{preset_name}'. Disponibili: {', '.join(PRESETS)}")
        self.preset = PRESETS[preset_name]
        self.preset_name = preset_name

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder(from_pretrained=False)
        self.timestamp = timestamp if timestamp is not None else self.preset.timestamp

        gallery_path = Path(gallery) if gallery else Path(self.preset.gallery)
        if not gallery_path.is_absolute():
            gallery_path = GALLERY_DIR / gallery_path
        self.gps_gallery = load_gps_data(str(gallery_path))
        self._initialize_gps_queue(queue_size)

        self.weights_folder = str(WEIGHTS_DIR)
        if from_pretrained:
            self._load_weights(epoch_num=epoch_num, epochs=epochs)

        self.device = "cpu"

    def to(self, device):
        self.device = str(device)
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _weight_names(self, epochs: bool) -> tuple[str, str, str]:
        if not self.timestamp:
            return ("image_encoder_mlp_weights.pth", "location_encoder_weights.pth", "logit_scale_weights.pth")
        prefix = "" if epochs else "fine_tuned_"
        return (
            f"{prefix}image_encoder_mlp_weights_{self.timestamp}.pth",
            f"{prefix}location_encoder_weights_{self.timestamp}.pth",
            f"{prefix}logit_scale_weights_{self.timestamp}.pth",
        )

    def _load_weights(self, epoch_num: int = 0, epochs: bool = False):
        folder = SNAPSHOTS_DIR / f"epoch_{epoch_num}" if epochs else WEIGHTS_DIR
        img_name, loc_name, scale_name = self._weight_names(epochs)
        paths = [folder / img_name, folder / loc_name, folder / scale_name]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Pesi GeoCLIP mancanti:\n  " + "\n  ".join(missing))

        log.info("GeoCLIP weights: %s", ", ".join(p.name for p in paths))
        self.image_encoder.mlp.load_state_dict(_torch_load(paths[0]))
        self.location_encoder.load_state_dict(_torch_load(paths[1]))
        self.logit_scale = nn.Parameter(_torch_load(paths[2]))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """Update the GPS queue.

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)

        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()

    def forward(self, image, location):
        """GeoCLIP's forward pass.

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            torch.Tensor: logits per image of shape (n, m)
        """
        image_features = F.normalize(self.image_encoder(image), dim=1)
        location_features = F.normalize(self.location_encoder(location), dim=1)
        return self.logit_scale.exp() * (image_features @ location_features.t())

    @torch.no_grad()
    def predict(self, image: ImageInput, top_k: int = 5):
        """Predict the ``top_k`` most likely GPS coordinates for an image.

        Args:
            image: path, ``PIL.Image`` or raw bytes.
            top_k: number of predictions to return.

        Returns:
            (top_pred_gps, top_pred_prob): tensors of shape (k, 2) and (k,).
        """
        pixel_values = self.image_encoder.preprocess_image(_open_image(image)).to(self.device)
        gps_gallery = self.gps_gallery.to(self.device)

        probs = self.forward(pixel_values, gps_gallery).softmax(dim=-1).cpu()
        top_k = max(1, min(int(top_k), probs.shape[1]))
        top_pred = torch.topk(probs, top_k, dim=1)
        return self.gps_gallery[top_pred.indices[0]], top_pred.values[0]
