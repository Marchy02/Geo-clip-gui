"""Lazy ML engines (GeoCLIP, EasyOCR, YOLO)."""

from __future__ import annotations

from .base import EngineStatus, EngineUnavailable


def statuses() -> list[EngineStatus]:
    from . import geoclip_engine, ocr_engine, yolo_engine

    return [geoclip_engine.ENGINE.status(), ocr_engine.ENGINE.status(), yolo_engine.ENGINE.status()]


__all__ = ["EngineStatus", "EngineUnavailable", "statuses"]
