"""Metadata removal (OPSEC): re-encode pixels only, dropping EXIF/XMP/GPS."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def strip_metadata(src: str | Path, dst: str | Path, quality: int = 95) -> Path:
    """Write a copy of ``src`` with no metadata.

    The EXIF orientation is applied to the pixels first, so the clean copy
    is displayed the same way as the original.
    """
    src, dst = Path(src), Path(dst)
    with Image.open(src) as img:
        img.load()
        fmt = img.format or "PNG"
        clean = ImageOps.exif_transpose(img)
        # Rebuild from raw pixels so no info dict / chunks are carried over.
        clean = Image.frombytes(clean.mode, clean.size, clean.tobytes())
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt.upper() in {"JPEG", "MPO"}:
        clean.convert("RGB").save(dst, "JPEG", quality=quality, optimize=True)
    elif fmt.upper() == "WEBP":
        clean.save(dst, "WEBP", quality=quality)
    else:
        if clean.mode == "P":
            clean = clean.convert("RGBA")
        clean.save(dst, "PNG", optimize=True)
    return dst
