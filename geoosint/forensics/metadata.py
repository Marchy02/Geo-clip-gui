"""Full metadata extraction: EXIF, GPS, XMP, PNG text chunks, ICC, thumbnails.

Also produces human readable *findings* (privacy leaks, editing software,
date inconsistencies, signs of social-network recompression) and
*AI-generation markers* (C2PA, IPTC DigitalSourceType, Stable Diffusion /
ComfyUI prompts, generator names in the Software tag).
"""

from __future__ import annotations

import mimetypes
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from ..models import GpsInfo, MetadataReport
from .jpeg import estimate_jpeg_quality

_MAX_VALUE_LEN = 200

# fmt: off
EDITING_SOFTWARE = (
    "photoshop", "lightroom", "gimp", "snapseed", "affinity", "pixelmator", "capture one", "picsart", "canva",
    "facetune", "vsco", "luminar", "paint.net", "darktable", "rawtherapee",
)
# fmt: on
# fmt: off
AI_GENERATORS = (
    "midjourney", "dall-e", "dall·e", "stable diffusion", "stablediffusion", "firefly", "imagen", "google ai",
    "gemini", "bing image creator", "microsoft designer", "leonardo", "flux", "novelai", "comfyui",
    "automatic1111", "invokeai", "ideogram", "runway", "chatgpt", "openai",
)
# fmt: on
AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia", "algorithmicmedia")
SERIAL_TAGS = ("BodySerialNumber", "LensSerialNumber", "SerialNumber", "CameraSerialNumber", "ImageUniqueID")
OWNER_TAGS = ("Artist", "Copyright", "CameraOwnerName", "OwnerName", "XPAuthor")
# Longest side typically produced by social networks / messengers after recompression.
SOCIAL_SIZES = {1080: "Instagram", 1280: "WhatsApp/Telegram", 1600: "WhatsApp HD", 2048: "Facebook/X"}


def _to_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:  # (num, den) legacy rational
            return float(Fraction(int(value[0]), int(value[1]))) if value[1] else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def dms_to_decimal(dms: Any, ref: str | None) -> float | None:
    """Convert EXIF (deg, min, sec) rationals to signed decimal degrees."""
    try:
        parts = [_to_float(v) for v in dms]
    except TypeError:
        return None
    if not parts or any(p is None for p in parts):
        return None
    while len(parts) < 3:
        parts.append(0.0)
    value = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if ref and ref.strip().upper() in {"S", "W"}:
        value = -value
    return round(value, 7)


def _clean(value: Any) -> str:
    if isinstance(value, bytes):
        # XP* tags are UTF-16LE, others are usually ASCII.
        for enc in ("utf-16-le", "utf-8", "latin-1") if value.count(b"\x00") > len(value) // 4 else ("utf-8", "latin-1"):
            try:
                text = value.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover
            text = repr(value)
        value = text.replace("\x00", "")
        if sum(ch.isprintable() for ch in value) < len(value) * 0.8:
            return f"<{len(text)} byte binari>"
    if isinstance(value, tuple):
        value = ", ".join(_clean(v) for v in value)
    text = str(value).strip()
    return text if len(text) <= _MAX_VALUE_LEN else text[:_MAX_VALUE_LEN] + "…"


def _parse_gps(gps_ifd: dict) -> GpsInfo | None:
    named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    lat = dms_to_decimal(named.get("GPSLatitude"), named.get("GPSLatitudeRef")) if "GPSLatitude" in named else None
    lon = dms_to_decimal(named.get("GPSLongitude"), named.get("GPSLongitudeRef")) if "GPSLongitude" in named else None
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None

    altitude = _to_float(named.get("GPSAltitude"))
    if altitude is not None and named.get("GPSAltitudeRef") in (1, b"\x01"):
        altitude = -altitude

    timestamp = None
    if "GPSDateStamp" in named and "GPSTimeStamp" in named:
        try:
            h, m, s = (_to_float(v) or 0 for v in named["GPSTimeStamp"])
            timestamp = f"{_clean(named['GPSDateStamp']).replace(':', '-')}T{int(h):02d}:{int(m):02d}:{int(s):02d}Z"
        except (TypeError, ValueError):
            timestamp = None

    return GpsInfo(
        lat=lat,
        lon=lon,
        altitude_m=round(altitude, 1) if altitude is not None else None,
        direction_deg=_to_float(named.get("GPSImgDirection")),
        direction_ref=_clean(named.get("GPSImgDirectionRef")) if named.get("GPSImgDirectionRef") else None,
        speed=_to_float(named.get("GPSSpeed")),
        timestamp_utc=timestamp,
        map_datum=_clean(named.get("GPSMapDatum")) if named.get("GPSMapDatum") else None,
    )


_XMP_ATTR = re.compile(r"([A-Za-z][\w-]*:[A-Za-z][\w-]*)\s*=\s*\"([^\"]*)\"")
_XMP_ELEM = re.compile(r"<([A-Za-z][\w-]*:[A-Za-z][\w-]*)>([^<]+)</\1>")
_XMP_SKIP_NS = ("xmlns", "rdf:", "x:", "stEvt:", "stRef:")


def parse_xmp(raw: bytes | str | None) -> dict[str, str]:
    """Flatten an XMP packet into ``prefix:name -> value`` (best effort, no XML deps)."""
    if not raw:
        return {}
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
    out: dict[str, str] = {}
    for key, value in [*_XMP_ATTR.findall(text), *_XMP_ELEM.findall(text)]:
        if key.startswith(_XMP_SKIP_NS) or not value.strip():
            continue
        out.setdefault(key, _clean(value))
    return out


def _find_raw_xmp(data: bytes) -> bytes | None:
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        return None
    end = data.find(b"</x:xmpmeta>", start)
    return data[start : end + 12] if end > 0 else None


def extract_metadata(path: str | Path) -> MetadataReport:
    path = Path(path)
    raw_bytes = path.read_bytes()
    report = MetadataReport(
        file_name=path.name,
        file_size=len(raw_bytes),
        mime=mimetypes.guess_type(path.name)[0] or "",
    )

    with Image.open(path) as img:
        report.format = img.format or ""
        report.mime = Image.MIME.get(img.format or "", report.mime)
        report.width, report.height = img.size
        report.mode = img.mode
        report.jpeg_quality = estimate_jpeg_quality(img)
        icc = img.info.get("icc_profile")
        if icc:
            report.icc_profile = _icc_description(icc)

        exif = img.getexif()
        tags: dict[str, Any] = {}
        for k, v in exif.items():
            tags[ExifTags.TAGS.get(k, f"Tag_{k:#06x}")] = v
        for ifd in (ExifTags.IFD.Exif, ExifTags.IFD.Interop):
            try:
                for k, v in exif.get_ifd(ifd).items():
                    tags[ExifTags.TAGS.get(k, f"Tag_{k:#06x}")] = v
            except (KeyError, ValueError, TypeError):
                continue
        tags.pop("ExifOffset", None)
        tags.pop("GPSInfo", None)
        tags.pop("MakerNote", None)  # vendor binary blob
        report.exif = {k: _clean(v) for k, v in sorted(tags.items())}

        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (KeyError, ValueError, TypeError):
            gps_ifd = {}
        report.gps = _parse_gps(gps_ifd) if gps_ifd else None

        try:
            report.has_thumbnail = bool(exif.get_ifd(ExifTags.IFD.IFD1))
        except (KeyError, ValueError, TypeError, AttributeError):
            report.has_thumbnail = False

        xmp_raw = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp") or _find_raw_xmp(raw_bytes)
        report.xmp = parse_xmp(xmp_raw)

        report.png_text = {
            str(k): _clean(v) for k, v in img.info.items() if isinstance(v, str) and k not in {"xmp", "XML:com.adobe.xmp", "icc_profile"}
        }

    make = report.exif.get("Make", "")
    model = report.exif.get("Model", "")
    report.camera = (model if model.lower().startswith(make.lower()) else f"{make} {model}").strip()
    report.lens = report.exif.get("LensModel", "") or report.xmp.get("aux:Lens", "")
    report.software = report.exif.get("Software", "") or report.xmp.get("xmp:CreatorTool", "")
    report.datetime_original = (
        report.exif.get("DateTimeOriginal")
        or report.exif.get("DateTimeDigitized")
        or report.xmp.get("exif:DateTimeOriginal")
        or report.xmp.get("photoshop:DateCreated")
        or report.exif.get("DateTime")
    )
    report.timezone_offset = report.exif.get("OffsetTimeOriginal") or report.exif.get("OffsetTime")

    report.ai_markers = _ai_markers(report, raw_bytes)
    report.findings = _findings(report)
    return report


def _icc_description(icc: bytes) -> str:
    try:
        from PIL import ImageCms

        profile = ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc))
        return ImageCms.getProfileDescription(profile).strip()
    except Exception:
        return f"profilo ICC ({len(icc)} byte)"


def _ai_markers(report: MetadataReport, raw: bytes) -> list[str]:
    markers: list[str] = []
    lowered = raw[:4_000_000].lower()
    if b"c2pa" in lowered or b"jumbf" in lowered:
        markers.append("Manifest C2PA / Content Credentials presente (verificare su contentcredentials.org/verify)")
    source_type = " ".join(v for k, v in report.xmp.items() if "digitalsourcetype" in k.lower()).lower()
    if any(t in source_type for t in AI_SOURCE_TYPES):
        markers.append(f"IPTC DigitalSourceType indica media generato da IA ({source_type.rsplit('/', 1)[-1]})")
    for key in ("parameters", "prompt", "workflow", "Dream", "sd-metadata", "invokeai_metadata"):
        if key in report.png_text:
            markers.append(f"Chunk PNG '{key}' tipico di Stable Diffusion/ComfyUI")
    haystack = " ".join([report.software, report.exif.get("ImageDescription", ""), report.exif.get("Artist", "")])
    haystack += " " + " ".join(report.xmp.values())
    for name in AI_GENERATORS:
        if name in haystack.lower():
            markers.append(f"Riferimento a generatore IA nei metadati: '{name}'")
            break
    return markers


def _findings(report: MetadataReport) -> list[str]:
    out: list[str] = []
    if not report.exif and not report.xmp:
        out.append("Nessun metadato EXIF/XMP: immagine probabilmente ripulita (social network, messaggistica o tool di sanitizzazione).")
    if report.gps:
        out.append(
            f"Coordinate GPS presenti nei metadati ({report.gps.lat:.6f}, {report.gps.lon:.6f}): prova di posizione forte, ma falsificabile."
        )
    software = report.software.lower()
    for name in EDITING_SOFTWARE:
        if name in software:
            out.append(f"Immagine elaborata con software di editing: {report.software}.")
            break
    original, modified = report.exif.get("DateTimeOriginal"), report.exif.get("DateTime")
    if original and modified and original != modified:
        out.append(f"Data di modifica ({modified}) diversa dalla data di scatto ({original}): file modificato dopo lo scatto.")
    for tag in SERIAL_TAGS:
        if report.exif.get(tag):
            out.append(f"Numero di serie esposto ({tag} = {report.exif[tag]}): collegabile ad altre foto dello stesso dispositivo.")
    for tag in OWNER_TAGS:
        if report.exif.get(tag):
            out.append(f"Dato identificativo dell'autore ({tag} = {report.exif[tag]}).")
    longest = max(report.width, report.height)
    if longest in SOCIAL_SIZES and not report.exif:
        out.append(f"Lato lungo {longest}px senza EXIF: compatibile con ricompressione {SOCIAL_SIZES[longest]}.")
    if report.jpeg_quality is not None and report.jpeg_quality < 80:
        out.append(f"Qualità JPEG stimata bassa ({report.jpeg_quality}): l'immagine è stata ricompressa almeno una volta.")
    if report.has_thumbnail and report.software:
        out.append("Miniatura EXIF presente in un file editato: confrontarla con l'immagine principale (può mostrare l'originale).")
    return out
