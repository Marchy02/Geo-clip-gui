"""GeoCLIP wrapper: top-k GPS predictions for an image."""

from __future__ import annotations

from ..models import GeoPoint
from .base import LazyEngine


def _load(preset: str):
    from geoclip import GeoCLIP, default_device

    model = GeoCLIP(preset=preset)
    model.eval()
    return model.to(default_device())


ENGINE = LazyEngine("GeoCLIP", ("torch", "transformers", "pandas"), _load)


def predict(image_path: str, top_k: int = 5, preset: str = "world") -> list[GeoPoint]:
    model = ENGINE.get(preset)
    gps, probs = model.predict(image_path, top_k=top_k)
    return [
        GeoPoint(lat=float(g[0]), lon=float(g[1]), prob=float(p), label=f"GeoCLIP #{i + 1}")
        for i, (g, p) in enumerate(zip(gps.tolist(), probs.tolist()))
    ]


def presets() -> dict[str, str]:
    """Preset name -> description, without importing torch."""
    return {
        "world": "Mondo — modello originale GeoCLIP (MP-16, 100K punti)",
        "ukraine_russia": "Ucraina/Russia — modello fine-tuned regionale",
    }
