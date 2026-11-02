"""Pivot links: maps, street-level imagery, satellite, reverse image search, text search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote, quote_plus


@dataclass(frozen=True)
class Link:
    name: str
    url: str
    category: str
    note: str = ""


def map_links(lat: float, lon: float, zoom: int = 17, when: datetime | None = None) -> list[Link]:
    ll = f"{lat:.6f},{lon:.6f}"
    sun_when = (when or datetime.now()).strftime("%Y.%m.%d/%H:%M")
    return [
        Link("Google Maps", f"https://www.google.com/maps/search/?api=1&query={ll}", "Mappe"),
        Link("OpenStreetMap", f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map={zoom}/{lat:.6f}/{lon:.6f}", "Mappe"),
        Link("Bing Maps", f"https://www.bing.com/maps?cp={lat:.6f}~{lon:.6f}&lvl={zoom}&style=h", "Mappe"),
        Link("Apple Maps", f"https://maps.apple.com/?ll={ll}&q=Pin&t=h", "Mappe"),
        Link("Yandex Maps", f"https://yandex.com/maps/?ll={lon:.6f},{lat:.6f}&z={zoom}&l=sat,skl", "Mappe", "ottimo per Est Europa/Russia"),
        Link("Baidu Maps", f"https://api.map.baidu.com/marker?location={ll}&coord_type=wgs84&output=html", "Mappe", "Cina"),
        Link("Google Street View", f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={ll}", "Street-level"),
        Link("Mapillary", f"https://www.mapillary.com/app/?lat={lat:.6f}&lng={lon:.6f}&z={zoom}", "Street-level", "foto crowdsourced"),
        Link("KartaView", f"https://kartaview.org/map/@{lat:.6f},{lon:.6f},{zoom}z", "Street-level"),
        Link("Google Earth", f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},300a,1200d,35y,0h,45t,0r", "Satellite / 3D"),
        Link(
            "Sentinel Hub EO Browser",
            f"https://apps.sentinel-hub.com/eo-browser/?zoom=14&lat={lat:.6f}&lng={lon:.6f}",
            "Satellite / 3D",
            "immagini storiche Sentinel/Landsat",
        ),
        Link(
            "Esri Wayback",
            f"https://livingatlas.arcgis.com/wayback/#mapCenter={lon:.6f}%2C{lat:.6f}%2C{zoom}",
            "Satellite / 3D",
            "archivio storico satellitare",
        ),
        Link("Zoom Earth", f"https://zoom.earth/maps/satellite/#view={ll},{min(zoom, 16)}z", "Satellite / 3D"),
        Link("Overpass Turbo", f"https://overpass-turbo.eu/?C={lat:.6f};{lon:.6f};{zoom}", "Analisi", "query sui dati OSM"),
        Link("Wikimapia", f"https://wikimapia.org/#lat={lat:.6f}&lon={lon:.6f}&z={zoom}&m=w", "Analisi"),
        Link("Wikipedia nei dintorni", f"https://en.wikipedia.org/wiki/Special:Nearby#/coord/{ll}", "Analisi"),
        Link(
            "GeoHack", f"https://geohack.toolforge.org/geohack.php?params={lat:.6f};{lon:.6f}", "Analisi", "decine di servizi in un click"
        ),
        Link(
            "PeakVisor (montagne)",
            f"https://peakvisor.com/panorama.html?lat={lat:.6f}&lng={lon:.6f}",
            "Analisi",
            "profilo dei monti all'orizzonte",
        ),
        Link("SunCalc", f"https://www.suncalc.org/#/{ll},{zoom}/{sun_when}/1/3", "Cronolocalizzazione"),
        Link("Windy (meteo)", f"https://www.windy.com/{lat:.3f}/{lon:.3f}?{lat:.3f},{lon:.3f},10", "Cronolocalizzazione"),
    ]


# Engines that only accept a manual upload when the image is local.
REVERSE_IMAGE_UPLOAD = [
    Link("Google Lens", "https://lens.google.com/", "Ricerca inversa", "migliore su luoghi e prodotti"),
    Link("Yandex Images", "https://yandex.com/images/", "Ricerca inversa", "migliore su volti, edifici, Est Europa"),
    Link("Bing Visual Search", "https://www.bing.com/visualsearch", "Ricerca inversa"),
    Link("TinEye", "https://tineye.com/", "Ricerca inversa", "trova la copia più vecchia / originale"),
    Link("Baidu Images", "https://graph.baidu.com/pcpage/index?tpl_from=pc", "Ricerca inversa", "Asia"),
    Link("Sogou", "https://pic.sogou.com/", "Ricerca inversa", "Cina"),
    Link("PimEyes", "https://pimeyes.com/en", "Ricerca inversa", "solo volti — rispettare la normativa privacy"),
]


def reverse_image_links(image_url: str | None = None) -> list[Link]:
    """Reverse image search links: direct searches if a public URL is known, upload pages otherwise."""
    if not image_url:
        return list(REVERSE_IMAGE_UPLOAD)
    u = quote(image_url, safe="")
    return [
        Link("Google Lens", f"https://lens.google.com/uploadbyurl?url={u}", "Ricerca inversa"),
        Link("Yandex Images", f"https://yandex.com/images/search?rpt=imageview&url={u}", "Ricerca inversa"),
        Link("Bing Visual Search", f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{u}", "Ricerca inversa"),
        Link("TinEye", f"https://tineye.com/search?url={u}", "Ricerca inversa"),
        Link("Baidu Images", f"https://graph.baidu.com/details?isfromtusoupc=1&tn=pc&carousel=0&image={u}", "Ricerca inversa"),
        Link("SauceNAO", f"https://saucenao.com/search.php?url={u}", "Ricerca inversa", "illustrazioni/anime"),
        *[link for link in REVERSE_IMAGE_UPLOAD if link.name in {"Sogou", "PimEyes"}],
    ]


def text_search_links(text: str) -> list[Link]:
    q = quote_plus(f'"{text}"')
    plain = quote(text)
    return [
        Link("Google", f"https://www.google.com/search?q={q}", "Testo"),
        Link("Google Maps", f"https://www.google.com/maps/search/{plain}", "Testo"),
        Link("Yandex", f"https://yandex.com/search/?text={q}", "Testo"),
        Link("Bing", f"https://www.bing.com/search?q={q}", "Testo"),
        Link("OSM Nominatim", f"https://nominatim.openstreetmap.org/ui/search.html?q={quote_plus(text)}", "Testo"),
        Link("Traduci", f"https://translate.google.com/?sl=auto&tl=it&text={plain}", "Testo"),
    ]


VERIFICATION_TOOLS = [
    Link("Content Credentials Verify", "https://contentcredentials.org/verify", "Verifica", "controllo C2PA"),
    Link("FotoForensics", "https://fotoforensics.com/", "Verifica", "ELA online"),
    Link("Forensically", "https://29a.ch/photo-forensics/", "Verifica", "clone detection, noise, ELA"),
    Link(
        "InVID / WeVerify",
        "https://www.invid-project.eu/tools-and-services/invid-verification-plugin/",
        "Verifica",
        "plugin per video e immagini",
    ),
    Link("Jeffrey's Exif Viewer", "https://exifinfo.org/", "Verifica"),
]
