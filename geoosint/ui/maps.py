"""Interactive Folium maps embedded as sandboxed iframes."""

from __future__ import annotations

import html
from dataclasses import dataclass

import folium
from folium import plugins

COLORS = {"final": "#ef4444", "exif": "#22c55e", "ai": "#f59e0b", "geoclip": "#06b6d4", "other": "#a78bfa"}


@dataclass
class Marker:
    lat: float
    lon: float
    label: str
    kind: str = "other"  # final | exif | ai | geoclip | other
    weight: float | None = None
    radius_km: float | None = None


def _base_map(center: tuple[float, float], zoom: int) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, © OpenStreetMap contributors",
        name="Scuro",
        max_zoom=16,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Stradale").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap (CC-BY-SA)",
        name="Topografica",
    ).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=", ", num_digits=6, prefix="Cursore:").add_to(m)
    return m


def render_map(markers: list[Marker], zoom: int | None = None, height: int = 460) -> str:
    if not markers:
        return empty_map(height)
    m = _base_map((markers[0].lat, markers[0].lon), zoom or 5)
    groups: dict[str, folium.FeatureGroup] = {}
    names = {"final": "Verdetto", "exif": "GPS EXIF", "ai": "Arbitro IA", "geoclip": "GeoCLIP", "other": "Punti"}
    for mk in markers:
        group = groups.get(mk.kind)
        if group is None:
            group = groups[mk.kind] = folium.FeatureGroup(name=names.get(mk.kind, mk.kind)).add_to(m)
        color = COLORS.get(mk.kind, COLORS["other"])
        tooltip = html.escape(mk.label)
        popup = folium.Popup(
            f"<b>{tooltip}</b><br><code>{mk.lat:.6f}, {mk.lon:.6f}</code><br>"
            f"<a href='https://www.google.com/maps/search/?api=1&query={mk.lat},{mk.lon}' target='_blank'>Google Maps</a> · "
            f"<a href='https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={mk.lat},{mk.lon}' target='_blank'>Street View</a>",
            max_width=280,
        )
        if mk.kind == "final":
            folium.Marker(
                [mk.lat, mk.lon], tooltip=tooltip, popup=popup, icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
            ).add_to(group)
        else:
            radius = 6 + 14 * min(1.0, (mk.weight or 0) * 3) if mk.weight is not None else 8
            folium.CircleMarker(
                [mk.lat, mk.lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                weight=2,
                tooltip=tooltip,
                popup=popup,
            ).add_to(group)
        if mk.radius_km:
            folium.Circle([mk.lat, mk.lon], radius=mk.radius_km * 1000, color=color, weight=1, fill=True, fill_opacity=0.05).add_to(group)
    folium.LayerControl(collapsed=True).add_to(m)
    if len(markers) > 1:
        lats = [mk.lat for mk in markers]
        lons = [mk.lon for mk in markers]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(40, 40), max_zoom=15)
    else:
        m.location = [markers[0].lat, markers[0].lon]
        m.options["zoom"] = zoom or 15
    return _iframe(m, height)


def empty_map(height: int = 460) -> str:
    m = _base_map((25, 10), 2)
    folium.LayerControl(collapsed=True).add_to(m)
    return _iframe(m, height)


def _iframe(m: folium.Map, height: int) -> str:
    doc = m.get_root().render()
    return (
        f'<iframe class="map-frame" style="height:{height}px" sandbox="allow-scripts allow-popups allow-same-origin" '
        f'srcdoc="{html.escape(doc, quote=True)}"></iframe>'
    )
