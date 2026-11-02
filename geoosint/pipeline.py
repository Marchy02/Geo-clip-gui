"""Analysis pipeline: runs every enabled step, isolates failures, fuses results.

Each step is independent: a missing model or a network error marks that
step as *skipped*/*error* in ``report.steps`` and the rest still runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import __version__
from .config import Settings, get_settings
from .engines.base import EngineUnavailable
from .geo.coords import haversine_km, weighted_centroid
from .models import AnalysisReport, FinalLocation, GeoPoint, LocationSource, StepStatus

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif", ".mpo"}


@dataclass
class AnalysisOptions:
    metadata: bool = True
    forensics: bool = True
    geoclip: bool = True
    ocr: bool = True
    yolo: bool = True
    ai: bool = True
    reverse_geocode: bool = True
    geoclip_preset: str | None = None
    top_k: int | None = None
    ocr_languages: list[str] | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    gemini_search: bool | None = None
    analyst_notes: str = ""


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


_register_heif()


def validate_image(path: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File non trovato: {p}")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_image_mb:
        raise ValueError(f"Immagine troppo grande ({size_mb:.1f} MB > {settings.max_image_mb} MB)")
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(f"File non riconosciuto come immagine valida: {exc}") from exc
    return p


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @contextmanager
    def _step(self, report: AnalysisReport, name: str) -> Iterator[StepStatus]:
        status = StepStatus(name=name, status="ok")
        start = time.perf_counter()
        try:
            yield status
        except EngineUnavailable as exc:
            status.status, status.detail = "skipped", str(exc)
            log.warning("Step %s saltato: %s", name, exc)
        except Exception as exc:
            status.status, status.detail = "error", f"{type(exc).__name__}: {exc}"
            log.exception("Step %s fallito", name)
        finally:
            status.duration_ms = int((time.perf_counter() - start) * 1000)
            report.steps.append(status)

    def run(self, image_path: str | Path, options: AnalysisOptions | None = None, progress: ProgressFn | None = None) -> AnalysisReport:
        opts = options or AnalysisOptions()
        s = self.settings
        path = validate_image(image_path, s)
        report = AnalysisReport(image_path=str(path), tool_version=__version__)
        img = str(path)

        def tick(frac: float, msg: str) -> None:
            if progress:
                progress(frac, msg)

        if opts.metadata:
            tick(0.05, "Metadati EXIF/XMP")
            with self._step(report, "Metadati"):
                from .forensics.metadata import extract_metadata

                report.metadata = extract_metadata(img)

        if opts.forensics:
            tick(0.12, "Analisi forense (hash, ELA, rumore)")
            with self._step(report, "Forense"):
                from .forensics.analysis import run_forensics

                report.forensics = run_forensics(img)

        if opts.geoclip:
            tick(0.2, "GeoCLIP: stima della posizione")
            with self._step(report, "GeoCLIP") as st:
                from .engines import geoclip_engine

                preset = opts.geoclip_preset or s.geoclip_preset
                report.geoclip = geoclip_engine.predict(img, top_k=opts.top_k or s.geoclip_top_k, preset=preset)
                if len(report.geoclip) > 1:
                    _, _, spread = weighted_centroid([(g.lat, g.lon, g.prob or 0) for g in report.geoclip])
                    report.geoclip_spread_km = round(spread, 1)
                st.detail = f"preset {preset}"

        if opts.ocr:
            tick(0.4, "OCR: lettura testi")
            with self._step(report, "OCR") as st:
                from .engines import ocr_engine

                langs = opts.ocr_languages or s.ocr_languages
                report.ocr = ocr_engine.read_text(img, langs, s.ocr_conf)
                st.detail = f"{len(report.ocr)} testi ({', '.join(langs)})"

        if opts.yolo:
            tick(0.55, "YOLO: rilevamento oggetti")
            with self._step(report, "YOLO") as st:
                from .engines import yolo_engine

                report.detections = yolo_engine.detect(img, s.yolo_model, s.yolo_conf)
                if report.detections:
                    report.detections_image = yolo_engine.draw(img, report.detections)
                st.detail = f"{len(report.detections)} oggetti"

        if opts.reverse_geocode and s.online_lookups and report.geoclip:
            tick(0.65, "Reverse geocoding (OpenStreetMap)")
            with self._step(report, "Reverse geocoding"):
                from .geo.geocode import reverse_geocode, short_label

                for g in report.geoclip[:3]:
                    g.address = short_label(reverse_geocode(g.lat, g.lon, zoom=10))

        if opts.ai:
            tick(0.72, "Arbitro IA (Gemini)")
            with self._step(report, "Arbitro IA") as st:
                from .ai import gemini

                key = opts.gemini_api_key or s.gemini_api_key
                if not key:
                    raise EngineUnavailable("Gemini: chiave API non configurata")
                model = opts.gemini_model or s.gemini_model
                context = gemini.build_context(report.geoclip, report.ocr, report.detections, report.metadata, opts.analyst_notes)
                report.ai = gemini.analyze(
                    img,
                    context,
                    api_key=key,
                    model=model,
                    use_search=s.gemini_search if opts.gemini_search is None else opts.gemini_search,
                    timeout_s=s.gemini_timeout_s,
                )
                if report.ai.error:
                    st.status, st.detail = "error", report.ai.error
                else:
                    st.detail = model

        tick(0.95, "Fusione dei risultati")
        report.final = fuse(report)
        if report.final and not report.final.point.address and s.online_lookups and opts.reverse_geocode:
            with self._step(report, "Indirizzo finale"):
                from .geo.geocode import reverse_geocode

                info = reverse_geocode(report.final.point.lat, report.final.point.lon, zoom=17)
                report.final.point.address = info.get("display_name", "")
        tick(1.0, "Completato")
        return report


def fuse(report: AnalysisReport) -> FinalLocation | None:
    """Pick the most trustworthy location and explain why."""
    gps = report.metadata.gps if report.metadata else None
    ai = report.ai
    top = report.geoclip[0] if report.geoclip else None

    if gps:
        note = "Coordinate GPS scritte dal dispositivo nei metadati EXIF."
        if ai and ai.lat is not None and ai.lon is not None:
            d = haversine_km(gps.lat, gps.lon, ai.lat, ai.lon)
            note += f" L'IA colloca l'immagine a {d:.1f} km" + (
                " (concordante)." if d < 25 else " — DISCORDANZA: possibile EXIF falsificato o foto riutilizzata."
            )
        return FinalLocation(GeoPoint(gps.lat, gps.lon, label="GPS EXIF"), LocationSource.EXIF, "alta", note)

    if ai and ai.has_location:
        level = "alta" if ai.confidence >= 75 else "media" if ai.confidence >= 40 else "bassa"
        name = ai.location_name or "Ipotesi IA"
        return FinalLocation(
            GeoPoint(ai.lat, ai.lon, prob=ai.confidence / 100, label=name),
            LocationSource.AI,
            level,
            ai.reasoning or "Località identificata dall'arbitro IA con prove visive indipendenti.",
        )

    if top:
        spread = report.geoclip_spread_km
        level = "media" if (top.prob or 0) > 0.2 and (spread or 0) < 300 else "bassa"
        note = f"Stima statistica GeoCLIP (probabilità {(top.prob or 0) * 100:.1f}%)."
        if spread is not None:
            note += f" Dispersione dei candidati: ~{spread:.0f} km."
        if ai and ai.country:
            note += f" L'IA suggerisce comunque il paese: {ai.country}."
        return FinalLocation(
            GeoPoint(top.lat, top.lon, prob=top.prob, label="GeoCLIP #1", address=top.address), LocationSource.GEOCLIP, level, note
        )

    if ai and ai.lat is not None and ai.lon is not None:
        return FinalLocation(
            GeoPoint(ai.lat, ai.lon, prob=ai.confidence / 100, label=ai.location_name or "Ipotesi IA"),
            LocationSource.AI,
            "bassa",
            "Ipotesi IA non confermata da prove indipendenti. " + ai.reasoning,
        )
    return None
