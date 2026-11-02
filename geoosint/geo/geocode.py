"""Forward and reverse geocoding through OpenStreetMap Nominatim.

Respects the Nominatim usage policy: identifying User-Agent, at most one
request per second, results cached in memory.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from functools import lru_cache
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config import get_settings

log = logging.getLogger(__name__)

NOMINATIM = "https://nominatim.openstreetmap.org"
_lock = threading.Lock()
_last_call = 0.0


class GeocodingError(RuntimeError):
    pass


def _get(path: str, params: dict) -> object:
    global _last_call
    settings = get_settings()
    if not settings.online_lookups:
        raise GeocodingError("ricerche online disattivate (ONLINE_LOOKUPS=false)")
    url = f"{NOMINATIM}/{path}?{urlencode({**params, 'format': 'jsonv2'})}"
    request = Request(url, headers={"User-Agent": settings.user_agent, "Accept-Language": "it,en"})
    with _lock:
        wait = 1.0 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            with urlopen(request, timeout=settings.http_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GeocodingError(f"Nominatim non raggiungibile: {exc}") from exc
        finally:
            _last_call = time.monotonic()


@lru_cache(maxsize=512)
def _reverse_cached(lat: float, lon: float, zoom: int) -> dict:
    data = _get("reverse", {"lat": lat, "lon": lon, "zoom": zoom, "addressdetails": 1})
    if not isinstance(data, dict) or "error" in data:
        return {}
    address = data.get("address", {})
    return {
        "display_name": data.get("display_name", ""),
        "country": address.get("country", ""),
        "country_code": (address.get("country_code") or "").upper(),
        "state": address.get("state", "") or address.get("region", ""),
        "city": address.get("city") or address.get("town") or address.get("village") or address.get("municipality", ""),
        "road": address.get("road", ""),
        "postcode": address.get("postcode", ""),
        "osm_url": f"https://www.openstreetmap.org/{data.get('osm_type', '')}/{data.get('osm_id', '')}" if data.get("osm_id") else "",
    }


def reverse_geocode(lat: float, lon: float, zoom: int = 14) -> dict:
    """Address for a point (rounded to ~10 m to improve cache hits)."""
    return _reverse_cached(round(lat, 4), round(lon, 4), zoom)


def short_label(info: dict) -> str:
    parts = [info.get("city"), info.get("state"), info.get("country")]
    return ", ".join(p for p in parts if p) or info.get("display_name", "")


@lru_cache(maxsize=256)
def search(query: str, limit: int = 5) -> list[dict]:
    data = _get("search", {"q": query, "limit": limit, "addressdetails": 0})
    if not isinstance(data, list):
        return []
    return [
        {
            "name": item.get("display_name", ""),
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "type": f"{item.get('category', '')}/{item.get('type', '')}",
        }
        for item in data
        if "lat" in item and "lon" in item
    ]
