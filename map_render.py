# map_render.py — Costruisce la mappa Folium e la serializza in HTML inline.

import base64
import io

import folium

import config
from analysis import GeoResult


def render_map_html(predictions: list[GeoResult]) -> str:
    """Restituisce un <iframe> HTML con la mappa Folium embedded in base64."""
    top = predictions[0]
    m = folium.Map(
        location=[top.lat, top.lon],
        zoom_start=config.MAP_ZOOM_START,
        tiles=config.MAP_TILES,
    )

    for i, geo in enumerate(predictions):
        folium.CircleMarker(
            location=[geo.lat, geo.lon],
            radius=config.MAP_MARKER_RADIUS if i == 0 else 6,
            color=config.MAP_MARKER_COLOR,
            fill=True,
            fill_opacity=1.0 if i == 0 else 0.4,
            tooltip=f"#{i+1}  {geo.prob*100:.1f}%  ({geo.lat:.4f}, {geo.lon:.4f})",
        ).add_to(m)

    buf = io.BytesIO()
    m.save(buf, close_file=False)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        'width="100%" height="450px" '
        'style="border:none; border-radius:12px; background:#080C14;">'
        "</iframe>"
    )
