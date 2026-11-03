"""EasyOCR wrapper: reads signs, shop names, plates and any visible text."""

from __future__ import annotations

from ..models import OcrText
from .base import LazyEngine


def _load(languages: tuple[str, ...]):
    import easyocr

    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        gpu = False
    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


ENGINE = LazyEngine("EasyOCR", ("easyocr",), _load)


def read_text(image_path: str, languages: list[str], min_conf: float = 0.2) -> list[OcrText]:
    reader = ENGINE.get(tuple(languages))
    results: list[OcrText] = []
    seen: set[str] = set()
    for box, text, conf in reader.readtext(image_path):
        clean = " ".join(str(text).split())
        if float(conf) < min_conf or not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        results.append(OcrText(text=clean, confidence=float(conf), box=[[float(x), float(y)] for x, y in box]))
    return sorted(results, key=lambda t: t.confidence, reverse=True)
