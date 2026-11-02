"""YOLO (ultralytics) wrapper: detects geolocation-relevant objects."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from ..models import Detection
from .base import LazyEngine

# COCO classes that carry geolocation value (driving side, infrastructure,
# vehicles, signage...).  They are highlighted in the UI and in the prompt.
# fmt: off
GEO_RELEVANT = {
    "car", "bus", "truck", "train", "motorcycle", "bicycle", "boat", "airplane",
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
}
# fmt: on

_PALETTE = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#facc15", "#fb7185"]


def _load(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


ENGINE = LazyEngine("YOLO", ("ultralytics",), _load)


def detect(image_path: str, model_path: str = "yolov8n.pt", min_conf: float = 0.35) -> list[Detection]:
    model = ENGINE.get(model_path)
    detections: list[Detection] = []
    for result in model(image_path, verbose=False):
        for box in result.boxes:
            conf = float(box.conf)
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(Detection(label=model.names[int(box.cls)], confidence=conf, box=(x1, y1, x2, y2)))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def summarize(detections: list[Detection]) -> dict[str, int]:
    """Label -> count, most frequent first."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def draw(image_path: str, detections: list[Detection]) -> Image.Image:
    """Return a copy of the image with labelled bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    draw_ctx = ImageDraw.Draw(img)
    width = max(2, round(max(img.size) / 400))
    try:
        font = ImageFont.load_default(size=max(12, round(max(img.size) / 60)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    labels = sorted({d.label for d in detections})
    for det in detections:
        if not det.box:
            continue
        color = _PALETTE[labels.index(det.label) % len(_PALETTE)]
        draw_ctx.rectangle(det.box, outline=color, width=width)
        caption = f"{det.label} {det.confidence:.0%}"
        x1, y1 = det.box[0], det.box[1]
        tb = draw_ctx.textbbox((x1, y1), caption, font=font)
        h = tb[3] - tb[1] + 4
        y_text = y1 - h if y1 - h > 0 else y1
        draw_ctx.rectangle((x1, y_text, x1 + (tb[2] - tb[0]) + 6, y_text + h), fill=color)
        draw_ctx.text((x1 + 3, y_text + 1), caption, fill="#0b1020", font=font)
    return img
