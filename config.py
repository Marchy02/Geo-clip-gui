# config.py — Parametri globali e variabili d'ambiente.
import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# --- GeoCLIP ---
GEOCLIP_TOP_K = 5

# --- YOLO ---
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35

# --- EasyOCR ---
OCR_LANGUAGES = ["en", "it", "de", "fr", "es"]
OCR_CONF_THRESHOLD = 0.2

# --- Mappa Folium ---
MAP_ZOOM_START = 6
MAP_TILES = "CartoDB dark_matter"
MAP_MARKER_COLOR = "#00FFB2"
MAP_MARKER_RADIUS = 10