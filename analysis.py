# analysis.py — Motore OSINT v4.0 (JSON Override)
from __future__ import annotations
import os
import json
import re
from PIL import Image, ExifTags
from typing import NamedTuple

from google import genai
from google.genai import types
from geoclip import GeoCLIP
from ultralytics import YOLO
import easyocr
import config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- STRUTTURE DATI ---
class GeoResult(NamedTuple):
    lat: float
    lon: float
    prob: float

class AnalysisResult(NamedTuple):
    geo_predictions: list[GeoResult]
    ocr_texts: list[str]
    yolo_objects: list[str]
    exif_info: dict[str, str]
    gemini_data: dict

# --- INIZIALIZZAZIONE ---
def _patch_geoclip(model: GeoCLIP) -> GeoCLIP:
    def _fix(module, args):
        x = args[0]
        if hasattr(x, "pooler_output"): return (x.pooler_output,)
        if isinstance(x, tuple): return (x[0],)
        return args
    model.image_encoder.mlp.register_forward_pre_hook(_fix)
    return model

def _init_gemini() -> genai.Client | None:
    if not config.GEMINI_API_KEY:
        return None
    try:
        return genai.Client(api_key=config.GEMINI_API_KEY)
    except Exception as e:
        print(f"Errore init Gemini Client: {e}")
        return None

print("Inizializzazione Motori OSINT in corso...")
_geoclip = _patch_geoclip(GeoCLIP())
_ocr     = easyocr.Reader(config.OCR_LANGUAGES)
_yolo    = YOLO(config.YOLO_MODEL_PATH)
_gemini_client = _init_gemini()

print("Modelli pronti." + (" [Gemini ✓]" if _gemini_client else " [Gemini ✗]"))

# --- FUNZIONI DI ESTRAZIONE ---
def extract_exif(img_path: str) -> dict[str, str]:
    try:
        raw = Image.open(img_path)._getexif() or {}
        return {
            ExifTags.TAGS[k]: str(v)[:80]
            for k, v in raw.items()
            if k in ExifTags.TAGS and ExifTags.TAGS[k] in {"Make", "Model", "DateTimeOriginal", "Software"}
        }
    except: 
        return {}

def run_ocr(img_path: str) -> list[str]:
    return [text for (_, text, prob) in _ocr.readtext(img_path) if prob > config.OCR_CONF_THRESHOLD]

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

def run_gemini(img_path: str, top: GeoResult, ocr_texts: list[str], yolo_objects: list[str]) -> dict:
    if _gemini_client is None:
        return {"override": False, "error": "Arbitro Gemini offline. Controlla la chiave."}
    
    try:
        prompt = f"""
        ANALISI OSINT.
        GeoCLIP propone: {top.lat}, {top.lon}
        OCR: {ocr_texts if ocr_texts else 'Nessuno'}
        YOLO: {yolo_objects if yolo_objects else 'Nessuno'}
        
        Identifica il luogo reale. Rispondi ESCLUSIVAMENTE con un JSON valido con questa struttura esatta:
        {{
            "override": true, 
            "lat": 45.4426, 
            "lon": 10.9972, 
            "location_name": "Nome esatto (es. Piazza delle Erbe, Verona, Italia)",
            "description": "Spiega cos'è questo luogo e indica i dettagli visivi esatti che confermano l'identità."
        }}
        Se non riconosci con certezza il posto, metti "override": false. Niente markdown. Solo JSON crudo.
        """
        
        response = _gemini_client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[prompt, Image.open(img_path)],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Pulizia stringa in caso Gemini aggiunga i backtick del markdown
        clean_json = re.sub(r"```json\n|```", "", response.text).strip()
        return json.loads(clean_json)
        
    except Exception as exc:
        return {"override": False, "error": f"Errore elaborazione Gemini: {exc}"}

# --- PIPELINE PRINCIPALE ---
def run_full_analysis(img_path: str) -> AnalysisResult:
    ocr_texts = run_ocr(img_path)
    yolo_objects = run_yolo(img_path)
    exif_info = extract_exif(img_path)
    geo_preds = run_geoclip(img_path)
    verdict_dict = run_gemini(img_path, geo_preds[0], ocr_texts, yolo_objects)
    
    return AnalysisResult(
        geo_predictions=geo_preds,
        ocr_texts=ocr_texts,
        yolo_objects=yolo_objects,
        exif_info=exif_info,
        gemini_data=verdict_dict
    )