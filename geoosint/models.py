"""Plain data structures shared by engines, pipeline, UI, CLI and reports."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class LocationSource(str, Enum):
    EXIF = "exif"
    AI = "ai"
    GEOCLIP = "geoclip"
    MANUAL = "manual"

    @property
    def label(self) -> str:
        return {
            "exif": "GPS nei metadati EXIF",
            "ai": "Arbitro IA (Gemini)",
            "geoclip": "GeoCLIP (stima statistica)",
            "manual": "Inserimento manuale",
        }[self.value]


@dataclass
class GeoPoint:
    lat: float
    lon: float
    prob: float | None = None
    label: str = ""
    address: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2


@dataclass
class OcrText:
    text: str
    confidence: float
    box: list[list[float]] | None = None


@dataclass
class GpsInfo:
    lat: float
    lon: float
    altitude_m: float | None = None
    direction_deg: float | None = None
    direction_ref: str | None = None
    speed: float | None = None
    timestamp_utc: str | None = None
    map_datum: str | None = None


@dataclass
class MetadataReport:
    file_name: str = ""
    file_size: int = 0
    format: str = ""
    mime: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    camera: str = ""
    lens: str = ""
    software: str = ""
    datetime_original: str | None = None
    timezone_offset: str | None = None
    gps: GpsInfo | None = None
    exif: dict[str, str] = field(default_factory=dict)
    xmp: dict[str, str] = field(default_factory=dict)
    png_text: dict[str, str] = field(default_factory=dict)
    icc_profile: str | None = None
    has_thumbnail: bool = False
    jpeg_quality: int | None = None
    findings: list[str] = field(default_factory=list)  # forensic/privacy observations
    ai_markers: list[str] = field(default_factory=list)


@dataclass
class Hashes:
    md5: str
    sha1: str
    sha256: str
    ahash: str = ""
    dhash: str = ""
    phash: str = ""


@dataclass
class ForensicsResult:
    hashes: Hashes | None = None
    ela_score: float | None = None
    ela_image: Any = None  # PIL.Image (not serialised)
    noise_image: Any = None  # PIL.Image (not serialised)
    findings: list[str] = field(default_factory=list)


@dataclass
class AiVerdict:
    override: bool = False
    lat: float | None = None
    lon: float | None = None
    location_name: str = ""
    country: str = ""
    confidence: int = 0  # 0-100
    reasoning: str = ""
    clues: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    model: str = ""
    error: str | None = None

    @property
    def has_location(self) -> bool:
        return self.override and self.lat is not None and self.lon is not None


@dataclass
class StepStatus:
    name: str
    status: str  # ok | skipped | error
    duration_ms: int = 0
    detail: str = ""


@dataclass
class FinalLocation:
    point: GeoPoint
    source: LocationSource
    confidence: str  # alta | media | bassa
    explanation: str


@dataclass
class AnalysisReport:
    image_path: str
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    metadata: MetadataReport | None = None
    forensics: ForensicsResult | None = None
    geoclip: list[GeoPoint] = field(default_factory=list)
    geoclip_spread_km: float | None = None
    ocr: list[OcrText] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    detections_image: Any = None  # PIL.Image with drawn boxes (not serialised)
    ai: AiVerdict | None = None
    final: FinalLocation | None = None
    steps: list[StepStatus] = field(default_factory=list)
    tool_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


_SKIP_FIELDS = {"ela_image", "noise_image", "detections_image"}


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums into JSON-serialisable values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.name not in _SKIP_FIELDS}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 7)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)
