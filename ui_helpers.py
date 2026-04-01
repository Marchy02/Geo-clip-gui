# ui_helpers.py — CSS e componenti HTML per la GUI Gradio.

from analysis import AnalysisResult, GeoResult

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
body, .gradio-container {
    background: #080C14 !important;
    color: #E2E8F0 !important;
    font-family: monospace;
}
.osint-card {
    background: #0D1421;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 5px;
    font-size: 0.8rem;
}
.verdict-box {
    background: #111827;
    border: 1px solid #00FFB2;
    border-radius: 10px;
    padding: 15px;
    color: #00FFB2;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 0 20px rgba(0,255,178,0.2);
}
.geo-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
.geo-table th { color: #00FFB2; border-bottom: 1px solid #1E293B; padding: 4px 8px; text-align: left; }
.geo-table td { padding: 4px 8px; border-bottom: 1px solid #0D1421; }
"""

# ---------------------------------------------------------------------------
# Componenti HTML
# ---------------------------------------------------------------------------

def _card(content: str) -> str:
    return f'<div class="osint-card">{content}</div>'


def render_geo_table(predictions: list[GeoResult]) -> str:
    rows = "".join(
        f"<tr><td>#{i+1}</td><td>{g.lat:.4f}</td><td>{g.lon:.4f}</td><td>{g.prob*100:.1f}%</td></tr>"
        for i, g in enumerate(predictions)
    )
    return _card(
        '<table class="geo-table">'
        "<thead><tr><th>#</th><th>Lat</th><th>Lon</th><th>Conf.</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_exif(exif_info: dict[str, str]) -> str:
    if not exif_info:
        return _card("Nessun metadato EXIF rilevante.")
    return _card("".join(f"<b>{k}:</b> {v}<br>" for k, v in exif_info.items()))


def render_ocr(texts: list[str]) -> str:
    return _card(", ".join(texts) if texts else "Nessun testo rilevato.")


def render_yolo(objects: list[str]) -> str:
    return _card(", ".join(objects) if objects else "Nessun oggetto rilevato.")


# ---------------------------------------------------------------------------
# Adattatore output → Gradio
# ---------------------------------------------------------------------------

def to_gradio_outputs(result: AnalysisResult, map_html: str) -> tuple:
    """
    Ordine atteso da app.py:
    (map_html, geo_table, exif_html, ocr_html, yolo_html, verdict_html)
    """
    return (
        map_html,
        render_geo_table(result.geo_predictions),
        render_exif(result.exif_info),
        render_ocr(result.ocr_texts),
        render_yolo(result.yolo_objects),
        f'<div class="verdict-box">{result.gemini_verdict}</div>',
    )
