# config.py — Parametri globali e variabili d'ambiente.
# Imposta la chiave Gemini con: export GEMINI_API_KEY="la_tua_chiave"
# oppure aggiungi GEMINI_API_KEY=... al file .env (già in .gitignore).

import os

# --- Gemini ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

# --- GeoCLIP ---
GEOCLIP_TOP_K: int = 5

# --- YOLO ---
YOLO_MODEL_PATH: str = "yolov8n.pt"
YOLO_CONF_THRESHOLD: float = 0.35

# --- EasyOCR ---
OCR_LANGUAGES: list[str] = ["en", "it", "de", "fr", "es"]
OCR_CONF_THRESHOLD: float = 0.2

# --- Mappa Folium ---
MAP_ZOOM_START: int = 6
MAP_TILES: str = "CartoDB dark_matter"
MAP_MARKER_COLOR: str = "#00FFB2"
MAP_MARKER_RADIUS: int = 10
