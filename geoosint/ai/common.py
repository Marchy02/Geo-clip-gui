"""Shared pieces of the AI arbiter: prompts, context building, image encoding, reply parsing.

Every provider (Gemini, Anthropic, OpenAI-compatible) sends the same prompt and
must answer with the same JSON object, so the verdict is comparable whichever
model produced it.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any

from PIL import Image

from ..models import AiVerdict, Detection, GeoPoint, MetadataReport, OcrText

MAX_SIDE = 2048

SYSTEM_PROMPT = """Sei un analista OSINT esperto di geolocalizzazione di immagini (stile GeoGuessr/Bellingcat).
Metodo: osserva l'immagine e i dati forniti, elenca gli indizi verificabili (lingua e alfabeto delle scritte,
insegne e nomi di attività, segnaletica stradale, targhe, lato di guida, architettura, vegetazione, clima,
pali e infrastrutture, orografia), poi formula l'ipotesi più specifica che le prove permettono.
Le coordinate di GeoCLIP sono solo una stima statistica: usale come indizio debole, MAI come prova.
Se hai lo strumento di ricerca web, usalo per verificare nomi di attività, landmark e testi letti.
Non inventare: se le prove non bastano per un luogo preciso, indica comunque paese/regione più probabili
con confidenza bassa e imposta "override" a false."""

RESPONSE_SPEC = """Rispondi SOLO con un oggetto JSON valido (nessun testo fuori dal JSON, niente markdown) con queste chiavi:
{
  "override": boolean,          // true SOLO se hai prove indipendenti (testo letto, landmark riconosciuto, ricerca web) per un punto preciso
  "lat": number | null,         // latitudine decimale del punto più probabile (anche se override=false, se stimabile)
  "lon": number | null,
  "location_name": string,      // es. "Piazza San Marco, Venezia"
  "country": string,            // paese più probabile
  "confidence": integer,        // 0-100
  "clues": [string],            // indizi osservati, uno per voce, brevi e verificabili
  "reasoning": string,          // ragionamento conciso che collega gli indizi alla conclusione
  "alternatives": [string]      // altre ipotesi plausibili, in ordine di probabilità
}
Scrivi i testi in italiano."""


def build_context(
    geoclip: list[GeoPoint],
    ocr: list[OcrText],
    detections: list[Detection],
    metadata: MetadataReport | None,
    analyst_notes: str = "",
) -> str:
    lines = ["DATI RACCOLTI DAGLI ALTRI MOTORI:"]
    if geoclip:
        lines.append("- Candidati GeoCLIP (stima statistica, poco affidabile):")
        for g in geoclip:
            where = f" — {g.address}" if g.address else ""
            lines.append(f"    {g.lat:.4f}, {g.lon:.4f} ({(g.prob or 0) * 100:.1f}%){where}")
    else:
        lines.append("- GeoCLIP: non eseguito.")
    texts = [f'"{t.text}" ({t.confidence:.0%})' for t in ocr[:40]]
    lines.append(f"- Testo letto via OCR: {', '.join(texts) if texts else 'nessuno'}")
    if detections:
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        lines.append("- Oggetti rilevati (YOLO): " + ", ".join(f"{k}×{v}" for k, v in counts.items()))
    else:
        lines.append("- Oggetti rilevati (YOLO): nessuno o non eseguito")
    if metadata:
        meta = []
        if metadata.camera:
            meta.append(f"fotocamera {metadata.camera}")
        if metadata.datetime_original:
            meta.append(f"data scatto {metadata.datetime_original} {metadata.timezone_offset or ''}".strip())
        if metadata.gps:
            meta.append(f"GPS EXIF {metadata.gps.lat:.6f}, {metadata.gps.lon:.6f}")
        if metadata.software:
            meta.append(f"software {metadata.software}")
        lines.append("- Metadati: " + ("; ".join(meta) if meta else "assenti"))
    if analyst_notes.strip():
        lines.append(f"- NOTE DELL'ANALISTA (contesto fornito dall'utente): {analyst_notes.strip()}")
    return "\n".join(lines)


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_verdict(text: str) -> dict[str, Any]:
    """Extract the first JSON object from the model's reply (tolerates fences/prose)."""
    if not text:
        raise ValueError("risposta vuota")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(cleaned)
        if not match:
            raise ValueError("nessun oggetto JSON nella risposta") from None
        return json.loads(match.group(0))


def _as_float(value: Any, lo: float, hi: float) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if lo <= f <= hi else None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def verdict_from_dict(data: dict[str, Any], model: str = "") -> AiVerdict:
    lat = _as_float(data.get("lat"), -90, 90)
    lon = _as_float(data.get("lon"), -180, 180)
    try:
        confidence = int(round(float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0
    override = bool(data.get("override")) and lat is not None and lon is not None
    return AiVerdict(
        override=override,
        lat=lat,
        lon=lon,
        location_name=str(data.get("location_name") or "").strip(),
        country=str(data.get("country") or "").strip(),
        confidence=max(0, min(100, confidence)),
        reasoning=str(data.get("reasoning") or data.get("description") or "").strip(),
        clues=_as_list(data.get("clues")),
        alternatives=_as_list(data.get("alternatives")),
        model=model,
    )


def encode_image(image_path: str) -> bytes:
    """Return the image as JPEG bytes, downscaled so the longest side is at most MAX_SIDE."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def redact(message: str, secret: str | None) -> str:
    """Remove an API key from an error message before it reaches the UI or the logs."""
    if secret and secret in message:
        message = message.replace(secret, "***")
    return message


def user_prompt(context: str) -> str:
    return context + "\n\n" + RESPONSE_SPEC


def verdict_from_text(text: str, model: str) -> AiVerdict:
    try:
        return verdict_from_dict(parse_verdict(text or ""), model=model)
    except (ValueError, json.JSONDecodeError) as exc:
        return AiVerdict(model=model, error=f"Risposta del modello non in formato JSON: {exc}")
