from .model import GeoCLIP, ImageEncoder, LocationEncoder
from .model.GeoCLIP import DEFAULT_PRESET, PRESETS, default_device

__all__ = ["GeoCLIP", "ImageEncoder", "LocationEncoder", "PRESETS", "DEFAULT_PRESET", "default_device", "train"]


def __getattr__(name):
    # Training utilities depend on wandb/tqdm: import them only when requested.
    if name == "train":
        from .train import train

        return train
    raise AttributeError(name)
