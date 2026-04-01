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
from google.genai import types

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
    
def _convert_to_degrees(value):
    """Converte le coordinate GPS da DMS a gradi decimali."""
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def extract_exif(img_path: str) -> dict[str, str]:
    exif_dict = {}
    try:
        img = Image.open(img_path)
        exif_raw = img._getexif()
        if not exif_raw:
            return {}

        # 1. Estrazione dati base
        for k, v in exif_raw.items():
            tag = ExifTags.TAGS.get(k, k)
            if tag in {"Make", "Model", "DateTimeOriginal", "Software"}:
                exif_dict[tag] = str(v)[:80]

        # 2. Estrazione GPS (Tag 34853)
        if 34853 in exif_raw:
            gps_info = {}
            for key in exif_raw[34853].keys():
                tag_name = ExifTags.GPSTAGS.get(key, key)
                gps_info[tag_name] = exif_raw[34853][key]

            # Calcolo Latitudine
            if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                lat = _convert_to_degrees(gps_info['GPSLatitude'])
                if gps_info['GPSLatitudeRef'] != 'N':
                    lat = -lat
                exif_dict['GPS_Lat'] = str(round(lat, 6))

            # Calcolo Longitudine
            if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                lon = _convert_to_degrees(gps_info['GPSLongitude'])
                if gps_info['GPSLongitudeRef'] != 'E':
                    lon = -lon
                exif_dict['GPS_Lon'] = str(round(lon, 6))

        return exif_dict
    except Exception as e:
        print(f"Errore parsing EXIF: {e}")
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
        SEI UN ANALISTA OSINT FORENSE.
        Dati rilevati:
        - GeoCLIP propone: {top.lat}, {top.lon} (ATTENZIONE: Questi dati sono inaffidabili. NON usarli per confermare l'identità del luogo senza prove visive indipendenti).
        - Testo OCR: {ocr_texts if ocr_texts else 'Nessun testo'}
        - Oggetti YOLO: {yolo_objects if yolo_objects else 'Nessun oggetto'}
        
        IL TUO COMPITO:
        Usa la ricerca web SOLO se ci sono testi OCR o landmark visivi chiari. 
        Se l'immagine è una mappa satellitare generica, rurale o senza dettagli unici, NON INVENTARE.
        
        Rispondi ESCLUSIVAMENTE con un JSON:
        {{
            "override": true, 
            "lat": 45.4426, 
            "lon": 10.9972, 
            "location_name": "Nome",
            "description": "Spiega le prove visive indipendenti trovate."
        }}
        
        REGOLA DI FERRO: Se non hai trovato una prova inconfutabile tramite OCR o dettagli architettonici unici, DEVI impostare "override": false. Non fare descrizioni basate solo sulle coordinate di GeoCLIP. NESSUN TESTO FUORI DAL JSON.
        """
        
        # --- CONFIGURAZIONE PULITA ---
        # Nessun riferimento a mime_type qui dentro. Solo il tool di ricerca e la temperatura bassa.
        my_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1
        )
        
        # Chiamata all'API
        response = _gemini_client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[prompt, Image.open(img_path)],
            config=my_config
        )
        
        # Pulizia brutale di eventuali formattazioni markdown che Gemini potrebbe aggiungere
        import re
        import json
        clean_json = re.sub(r"```json\n|```", "", response.text).strip()
        
        return json.loads(clean_json)
        
    except json.JSONDecodeError as e:
        return {"override": False, "error": f"Errore parsing JSON (Gemini ha scritto testo fuori formato): {e}"}
    except Exception as exc:
        return {"override": False, "error": f"Errore API Gemini: {exc}"}


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