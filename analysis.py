# analysis.py — Caricamento modelli e pipeline di analisi.
# Usa il nuovo SDK google-genai (non più google-generativeai, deprecato).

from __future__ import annotations

from PIL import Image, ExifTags
from typing import NamedTuple

from google import genai
from geoclip import GeoCLIP
from ultralytics import YOLO
import easyocr

# Carica .env se presente — il file deve avere la forma:
#   GEMINI_API_KEY=AIza...          ← SENZA "export"
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # funziona anche con `export GEMINI_API_KEY=...` nel terminale

import config


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------

class GeoResult(NamedTuple):
    lat: float
    lon: float
    prob: float  # 0.0–1.0


class AnalysisResult(NamedTuple):
    geo_predictions: list[GeoResult]
    ocr_texts: list[str]
    yolo_objects: list[str]
    exif_info: dict[str, str]
    gemini_verdict: str


# ---------------------------------------------------------------------------
# Inizializzazione modelli (una volta sola al primo import)
# ---------------------------------------------------------------------------

def _patch_geoclip(model: GeoCLIP) -> GeoCLIP:
    """Compatibilità con versioni recenti di transformers."""
    def _fix(module, args):
        x = args[0]
        if hasattr(x, "pooler_output"):
            return (x.pooler_output,)
        if isinstance(x, tuple):
            return (x[0],)
        return args
    model.image_encoder.mlp.register_forward_pre_hook(_fix)
    return model


def _init_gemini() -> genai.Client | None:
    """Crea il client Gemini con il nuovo SDK google-genai."""
    if not config.GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY non trovata.")
        print("   Assicurati che il file .env contenga:  GEMINI_API_KEY=AIza...")
        print("   (senza la parola 'export' davanti)")
        return None
    return genai.Client(api_key=config.GEMINI_API_KEY)


print("Caricamento modelli...")
_geoclip = _patch_geoclip(GeoCLIP())
_ocr     = easyocr.Reader(config.OCR_LANGUAGES)
_yolo    = YOLO(config.YOLO_MODEL_PATH)
_gemini  = _init_gemini()
print("Modelli pronti." + (" [Gemini ✓]" if _gemini else " [Gemini ✗]"))


# ---------------------------------------------------------------------------
# Analisi singoli step
# ---------------------------------------------------------------------------

_EXIF_TAGS_WANTED = {"Make", "Model", "DateTimeOriginal", "Software"}


def extract_exif(img_path: str) -> dict[str, str]:
    try:
        raw = Image.open(img_path)._getexif() or {}
        return {
            ExifTags.TAGS[k]: str(v)[:80]
            for k, v in raw.items()
            if k in ExifTags.TAGS and ExifTags.TAGS[k] in _EXIF_TAGS_WANTED
        }
    except Exception:
        return {}


def run_ocr(img_path: str) -> list[str]:
    return [
        text
        for (_, text, prob) in _ocr.readtext(img_path)
        if prob > config.OCR_CONF_THRESHOLD
    ]


def run_yolo(img_path: str) -> list[str]:
    return sorted({
        _yolo.names[int(box.cls)].upper()
        for result in _yolo(img_path, verbose=False)
        for box in result.boxes
        if float(box.conf) > config.YOLO_CONF_THRESHOLD
    })


def run_geoclip(img_path: str) -> list[GeoResult]:
    gps_list, prob_list = _geoclip.predict(img_path, top_k=config.GEOCLIP_TOP_K)
    return [
        GeoResult(lat=float(g[0]), lon=float(g[1]), prob=float(p))
        for g, p in zip(gps_list, prob_list)
    ]


_GEMINI_PROMPT = """\
ANALISI OSINT RICHIESTA.

GeoCLIP propone: {geoclip}
Testo rilevato (OCR): {ocr}
Oggetti in scena (YOLO): {yolo}

Identifica il luogo reale nella foto. Ignora i bias noti di GeoCLIP
(es. confusione Italia/Romania). Se riconosci un luogo specifico
(es. Arena di Verona, Colosseo, ecc.) indicalo esplicitamente e motiva.\
"""


def run_gemini(img_path: str, top: GeoResult,
               ocr_texts: list[str], yolo_objects: list[str]) -> str:
    if _gemini is None:
        return "⚠️ Arbitro Gemini offline — vedi istruzioni sopra nel terminale."
    try:
        prompt = _GEMINI_PROMPT.format(
            geoclip=f"Lat {top.lat:.4f}, Lon {top.lon:.4f} ({top.prob*100:.1f}%)",
            ocr=", ".join(ocr_texts) or "nessuno",
            yolo=", ".join(yolo_objects) or "nessuno",
        )
        # Nuovo SDK: client.models.generate_content
        response = _gemini.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[prompt, Image.open(img_path)],
        )
        return response.text
    except Exception as exc:
        return f"Errore Gemini: {exc}"


# ---------------------------------------------------------------------------
# Pipeline completa
# ---------------------------------------------------------------------------

def run_full_analysis(img_path: str) -> AnalysisResult:
    ocr_texts    = run_ocr(img_path)
    yolo_objects = run_yolo(img_path)
    exif_info    = extract_exif(img_path)
    geo_preds    = run_geoclip(img_path)
    verdict      = run_gemini(img_path, geo_preds[0], ocr_texts, yolo_objects)

    return AnalysisResult(
        geo_predictions=geo_preds,
        ocr_texts=ocr_texts,
        yolo_objects=yolo_objects,
        exif_info=exif_info,
        gemini_verdict=verdict,
    )