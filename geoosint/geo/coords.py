"""Coordinate parsing, formatting and conversions (pure Python, no deps).

Supported input formats (``parse_coordinates``):
    45.4408, 12.3155 · 45.4408 12.3155 · 45,4408; 12,3155
    45°26'27"N 12°18'56"E · N 45 26.45 E 12 18.93 · 45.44N 12.31O
    Google Maps / OSM / Apple / Bing URLs · geo:45.44,12.31
    Plus Code (8FVC9G8F+6X) · Geohash (u20fgg…) · UTM (33T 288420 5034640)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0088


class CoordinateError(ValueError):
    pass


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float

    def __post_init__(self):
        if not (-90 <= self.lat <= 90 and -180 <= self.lon <= 180):
            raise CoordinateError(f"Coordinate fuori range: {self.lat}, {self.lon}")

    def __str__(self) -> str:
        return f"{self.lat:.6f}, {self.lon:.6f}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def to_dms(lat: float, lon: float, precision: int = 1) -> str:
    def part(value: float, pos: str, neg: str) -> str:
        hemi = pos if value >= 0 else neg
        value = abs(value)
        d = int(value)
        m_full = (value - d) * 60
        m = int(m_full)
        s = round((m_full - m) * 60, precision)
        if s >= 60:
            s, m = 0.0, m + 1
        if m >= 60:
            m, d = 0, d + 1
        return f"{d}°{m:02d}'{s:0{3 + precision}.{precision}f}\"{hemi}"

    return f"{part(lat, 'N', 'S')} {part(lon, 'E', 'W')}"


def to_ddm(lat: float, lon: float) -> str:
    def part(value: float, pos: str, neg: str) -> str:
        hemi = pos if value >= 0 else neg
        value = abs(value)
        d = int(value)
        return f"{hemi} {d}° {(value - d) * 60:06.3f}'"

    return f"{part(lat, 'N', 'S')} {part(lon, 'E', 'W')}"


# ---------------------------------------------------------------------------
# Geohash
# ---------------------------------------------------------------------------

_GEOHASH = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int = 9) -> str:
    lat_rng, lon_rng = [-90.0, 90.0], [-180.0, 180.0]
    out, bits, bit_count, even = [], 0, 0, True
    while len(out) < precision:
        rng, val = (lon_rng, lon) if even else (lat_rng, lat)
        mid = (rng[0] + rng[1]) / 2
        if val >= mid:
            bits = (bits << 1) | 1
            rng[0] = mid
        else:
            bits <<= 1
            rng[1] = mid
        even = not even
        bit_count += 1
        if bit_count == 5:
            out.append(_GEOHASH[bits])
            bits, bit_count = 0, 0
    return "".join(out)


def geohash_decode(code: str) -> LatLon:
    lat_rng, lon_rng = [-90.0, 90.0], [-180.0, 180.0]
    even = True
    for ch in code.lower():
        idx = _GEOHASH.find(ch)
        if idx < 0:
            raise CoordinateError(f"Geohash non valido: {code}")
        for shift in range(4, -1, -1):
            rng = lon_rng if even else lat_rng
            mid = (rng[0] + rng[1]) / 2
            if (idx >> shift) & 1:
                rng[0] = mid
            else:
                rng[1] = mid
            even = not even
    return LatLon((lat_rng[0] + lat_rng[1]) / 2, (lon_rng[0] + lon_rng[1]) / 2)


# ---------------------------------------------------------------------------
# Open Location Code (Plus Codes) — full 10-digit codes
# ---------------------------------------------------------------------------

_OLC = "23456789CFGHJMPQRVWX"
_OLC_RES = 8000  # 20^3: resolution of a 10-digit code is 1/8000 degree


def plus_code_encode(lat: float, lon: float) -> str:
    lat = min(max(lat, -90.0), 90.0)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat_val = min(int(math.floor((lat + 90) * _OLC_RES)), 180 * _OLC_RES - 1)
    lon_val = int(math.floor((lon + 180) * _OLC_RES))
    digits: list[str] = []
    for _ in range(5):
        digits.append(_OLC[lon_val % 20])
        digits.append(_OLC[lat_val % 20])
        lat_val //= 20
        lon_val //= 20
    code = "".join(reversed(digits))
    return f"{code[:8]}+{code[8:]}"


def plus_code_decode(code: str) -> LatLon:
    clean = code.upper().replace("+", "").rstrip("0")
    if len(clean) < 2 or len(clean) % 2 or any(c not in _OLC for c in clean) or len(clean) > 10:
        raise CoordinateError(f"Plus Code non valido o abbreviato: {code}")
    lat = lon = 0.0
    res = 20.0
    for i in range(0, len(clean), 2):
        lat += _OLC.index(clean[i]) * res
        lon += _OLC.index(clean[i + 1]) * res
        last_res = res
        res /= 20
    return LatLon(round(lat - 90 + last_res / 2, 7), round(lon - 180 + last_res / 2, 7))


# ---------------------------------------------------------------------------
# UTM (WGS84)
# ---------------------------------------------------------------------------

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996
_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def utm_zone(lat: float, lon: float) -> int:
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32
    if 72 <= lat <= 84 and lon >= 0:
        for limit, zone in ((9, 31), (21, 33), (33, 35), (42, 37)):
            if lon < limit:
                return zone
    return int((lon + 180) // 6) % 60 + 1


def to_utm(lat: float, lon: float) -> str:
    if not -80 <= lat <= 84:
        return "n/d (fuori dal dominio UTM)"
    zone = utm_zone(lat, lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    phi, lam = math.radians(lat), math.radians(lon)
    n = _A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = _EP2 * math.cos(phi) ** 2
    a = math.cos(phi) * (lam - lon0)
    m = _A * (
        (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256) * phi
        - (3 * _E2 / 8 + 3 * _E2**2 / 32 + 45 * _E2**3 / 1024) * math.sin(2 * phi)
        + (15 * _E2**2 / 256 + 45 * _E2**3 / 1024) * math.sin(4 * phi)
        - (35 * _E2**3 / 3072) * math.sin(6 * phi)
    )
    easting = _K0 * n * (a + (1 - t + c) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * _EP2) * a**5 / 120) + 500000
    northing = _K0 * (
        m
        + n * math.tan(phi) * (a**2 / 2 + (5 - t + 9 * c + 4 * c**2) * a**4 / 24 + (61 - 58 * t + t**2 + 600 * c - 330 * _EP2) * a**6 / 720)
    )
    if lat < 0:
        northing += 10_000_000
    band = _BANDS[min(int((lat + 80) // 8), len(_BANDS) - 1)]
    return f"{zone}{band} {easting:.0f} {northing:.0f}"


def from_utm(zone: int, band: str, easting: float, northing: float) -> LatLon:
    northern = band.upper() >= "N"
    x = easting - 500000
    y = northing if northern else northing - 10_000_000
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    n1 = _A / math.sqrt(1 - _E2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = _EP2 * math.cos(phi1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720
    )
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120) / math.cos(
        phi1
    )
    return LatLon(round(math.degrees(lat), 7), round(math.degrees(lon), 7))


# ---------------------------------------------------------------------------
# Geodesy helpers
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> LatLon:
    delta = distance_km / EARTH_RADIUS_KM
    theta = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(delta) + math.cos(p1) * math.sin(delta) * math.cos(theta))
    l2 = l1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(p1), math.cos(delta) - math.sin(p1) * math.sin(p2))
    return LatLon(math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180)


def compass_point(bearing_deg: float) -> str:
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return names[int((bearing_deg % 360) / 22.5 + 0.5) % 16]


def weighted_centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Spherical weighted centroid of (lat, lon, weight) and the weighted mean distance (km)."""
    if not points:
        raise CoordinateError("Nessun punto")
    total = sum(max(w, 0.0) for _, _, w in points) or float(len(points))
    x = y = z = 0.0
    for lat, lon, w in points:
        w = (max(w, 0.0) or 1.0) / total
        p, lam = math.radians(lat), math.radians(lon)
        x += w * math.cos(p) * math.cos(lam)
        y += w * math.cos(p) * math.sin(lam)
        z += w * math.sin(p)
    lat_c = math.degrees(math.atan2(z, math.hypot(x, y)))
    lon_c = math.degrees(math.atan2(y, x))
    spread = sum(((max(w, 0.0) or 1.0) / total) * haversine_km(lat_c, lon_c, lat, lon) for lat, lon, w in points)
    return lat_c, lon_c, spread


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_URL_PATTERNS = [
    re.compile(r"!3d(" + _NUM + r")!4d(" + _NUM + r")"),  # Google place data
    re.compile(r"@(" + _NUM + r"),(" + _NUM + r")"),  # Google @lat,lon
    re.compile(r"[?&](?:q|ll|query|sll|center|daddr|destination)=(" + _NUM + r")(?:,|%2C)(" + _NUM + r")", re.I),
    re.compile(r"[?&]lat=(" + _NUM + r").*?[?&](?:lon|lng)=(" + _NUM + r")", re.I),
    re.compile(r"[#/]map=\d+/(" + _NUM + r")/(" + _NUM + r")"),  # OSM
    re.compile(r"cp=(" + _NUM + r")~(" + _NUM + r")"),  # Bing
    re.compile(r"^geo:(" + _NUM + r"),(" + _NUM + r")", re.I),
]
_UTM = re.compile(r"^\s*(\d{1,2})\s*([C-HJ-NP-X])\s+(\d{5,7}(?:\.\d+)?)\s*m?E?\s+(\d{6,8}(?:\.\d+)?)\s*m?N?\s*$", re.I)


def _f(value: str) -> float:
    return float(value.replace(",", "."))


def _parse_dms(text: str) -> LatLon | None:
    """DMS / DDM / DD with hemisphere letters, leading or trailing (O = Ovest)."""
    t = re.sub(r"(\d),(\d)", r"\1.\2", text.upper())
    if len(re.findall(r"[NSEWO]", t)) != 2:
        return None
    if re.match(r"\s*[NSEWO]", t):
        segments = [(hemi, body) for hemi, body in re.findall(r"([NSEWO])([^NSEWO]*)", t)]
    else:
        segments = [(hemi, body) for body, hemi in re.findall(r"([^NSEWO]*)([NSEWO])", t)]
    lat = lon = None
    for hemi, body in segments:
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", body)]
        if not 1 <= len(nums) <= 3:
            return None
        value = nums[0] + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
        if hemi in "NS":
            lat = -value if hemi == "S" else value
        else:
            lon = -value if hemi in "WO" else value
    if lat is None or lon is None:
        return None
    return LatLon(round(lat, 7), round(lon, 7))


def parse_coordinates(text: str) -> LatLon:
    """Parse almost any coordinate notation into decimal degrees."""
    raw = (text or "").strip()
    if not raw:
        raise CoordinateError("Inserisci delle coordinate")

    for pattern in _URL_PATTERNS:
        m = pattern.search(raw)
        if m:
            return LatLon(_f(m.group(1)), _f(m.group(2)))

    m = _UTM.match(raw)
    if m:
        return from_utm(int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)))

    compact = raw.replace(" ", "")
    if "+" in compact and re.fullmatch(r"[23456789CFGHJMPQRVWX0]{8}\+[23456789CFGHJMPQRVWX]{0,3}", compact, re.I):
        return plus_code_decode(compact)

    if re.fullmatch(r"[0-9bcdefghjkmnpqrstuvwxyz]{5,12}", compact) and not compact.isdigit():
        return geohash_decode(compact)

    # Decimal pair: "45.1, 12.2" / "45,1 12,2" / "45.1;12.2"
    m = re.fullmatch(r"\s*(" + _NUM + r")\s*[,;\s]\s*(" + _NUM + r")\s*", raw)
    if m:
        return LatLon(_f(m.group(1)), _f(m.group(2)))
    nums = re.findall(r"[-+]?\d+\.\d+", raw)
    if len(nums) == 2 and not re.search(r"[NSEWO°'\"]", raw, re.I):
        return LatLon(float(nums[0]), float(nums[1]))

    dms = _parse_dms(raw)
    if dms:
        return dms
    raise CoordinateError(f"Formato coordinate non riconosciuto: {raw!r}")


def all_formats(lat: float, lon: float) -> dict[str, str]:
    return {
        "Decimale": f"{lat:.6f}, {lon:.6f}",
        "DMS": to_dms(lat, lon),
        "DDM": to_ddm(lat, lon),
        "UTM (WGS84)": to_utm(lat, lon),
        "Plus Code": plus_code_encode(lat, lon),
        "Geohash": geohash_encode(lat, lon),
    }
