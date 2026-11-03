"""Interactive Folium maps embedded as sandboxed iframes."""

from __future__ import annotations

import html
from dataclasses import dataclass

import folium
from folium import plugins

# Same hues as the legend tokens, brightened to stay visible on satellite imagery.
COLORS = {"final": "#e0408f", "exif": "#3fbf7a", "ai": "#e09a4f", "geoclip": "#4aa8e0", "other": "#b39ddb"}


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
    # The first layer is the default: satellite imagery is what you compare a photo against.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Stradale").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap (CC-BY-SA)",
        name="Topografica",
        max_zoom=17,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, © OpenStreetMap contributors",
        name="Scuro",
        max_zoom=16,
    ).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=", ", num_digits=6, prefix="Cursore").add_to(m)
    return m


def render_map(markers: list[Marker], zoom: int | None = None, height: int = 460) -> str:
    if not markers:
        return empty_map(height)
    m = _base_map((markers[0].lat, markers[0].lon), zoom or 5)
    groups: dict[str, folium.FeatureGroup] = {}
    names = {"final": "Verdetto", "exif": "GPS EXIF", "ai": "Ipotesi IA", "geoclip": "Candidati GeoCLIP", "other": "Punti"}
    for mk in sorted(markers, key=lambda m: m.kind == "final"):  # verdict drawn last, on top
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
            # Crosshair: outer ring + solid core, drawn above everything else.
            folium.CircleMarker([mk.lat, mk.lon], radius=14, color=color, weight=3, fill=False).add_to(group)
            folium.CircleMarker(
                [mk.lat, mk.lon],
                radius=6,
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                tooltip=tooltip,
                popup=popup,
            ).add_to(group)
        else:
            radius = 5 + 12 * min(1.0, (mk.weight or 0) * 3) if mk.weight is not None else 8
            folium.CircleMarker(
                [mk.lat, mk.lon],
                radius=radius,
                color="#ffffff",
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1.5,
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
