"""Pixel-level forensic filters: Error Level Analysis and noise residual.

These are *indicators*, not proof: uniform ELA/noise means nothing was
pasted with a different compression history; bright isolated regions are
worth a closer look (splicing, clone stamp, inpainting).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from ..models import ForensicsResult
from .hashing import compute_hashes

MAX_SIDE = 1600  # keep filters fast on huge photos


def _load_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return img


def error_level_analysis(img: Image.Image, quality: int = 90) -> tuple[Image.Image, float]:
    """Re-save at ``quality`` and amplify the difference.

    Returns the ELA visualisation and the mean error level (0-255).
    """
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(img, resaved)
    arr = np.asarray(diff, dtype=np.float32)
    max_diff = float(arr.max()) or 1.0
    scaled = np.clip(arr * (255.0 / max_diff), 0, 255).astype(np.uint8)
    return Image.fromarray(scaled), float(arr.mean())


def noise_residual(img: Image.Image, amplification: float = 12.0) -> Image.Image:
    """High-pass noise map: image minus its median-filtered version."""
    gray = img.convert("L")
    smooth = gray.filter(ImageFilter.MedianFilter(3))
    residual = np.abs(np.asarray(gray, dtype=np.int16) - np.asarray(smooth, dtype=np.int16)).astype(np.float32)
    return Image.fromarray(np.clip(residual * amplification, 0, 255).astype(np.uint8))


def _ela_regional_contrast(ela: Image.Image, grid: int = 8) -> float:
    """Ratio between the brightest grid cell and the median cell of the ELA map."""
    arr = np.asarray(ela.convert("L"), dtype=np.float32)
    h, w = arr.shape
    cells = [arr[r * h // grid : (r + 1) * h // grid, c * w // grid : (c + 1) * w // grid].mean() for r in range(grid) for c in range(grid)]
    median = max(float(np.median(cells)), 2.0)  # floor: flat areas (sky, walls) have ~0 error
    return float(max(cells)) / median


def run_forensics(path: str | Path, ela_quality: int = 90) -> ForensicsResult:
    img = _load_rgb(path)
    ela_img, ela_score = error_level_analysis(img, ela_quality)
    result = ForensicsResult(
        hashes=compute_hashes(path),
        ela_score=round(ela_score, 3),
        ela_image=ela_img,
        noise_image=noise_residual(img),
    )
    contrast = _ela_regional_contrast(ela_img)
    if contrast > 4:
        result.findings.append(
            f"ELA: una regione ha un livello d'errore {contrast:.1f}× superiore alla mediana — "
            "possibile area incollata/ritoccata, verificare visivamente."
        )
    else:
        result.findings.append("ELA: livello d'errore abbastanza uniforme, nessuna anomalia evidente.")
    return result
