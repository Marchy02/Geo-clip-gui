"""Solar geometry for chronolocation (NOAA algorithm, accuracy ~0.01°).

* ``sun_position``: azimuth/elevation of the sun for a place and instant.
* ``sun_times``: sunrise, solar noon, sunset, golden hours.
* ``times_for_shadow``: given an object's height and its shadow length
  (and optionally the shadow direction), find the times of day that match
  — the classic shadow-based chronolocation technique.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception


@dataclass(frozen=True)
class SunPosition:
    when: datetime
    azimuth: float  # degrees from true north, clockwise
    elevation: float  # degrees above the horizon (refraction-corrected)

    @property
    def shadow_azimuth(self) -> float:
        return (self.azimuth + 180) % 360

    @property
    def shadow_ratio(self) -> float | None:
        """Shadow length / object height (None when the sun is below the horizon)."""
        if self.elevation <= 0.1:
            return None
        return 1 / math.tan(math.radians(self.elevation))


def parse_tz(value: str | None) -> tzinfo:
    """Accept IANA names (Europe/Rome), 'UTC', or offsets like '+02:00' / '-5'."""
    text = (value or "UTC").strip()
    if text.upper() in {"UTC", "Z", "GMT"}:
        return timezone.utc
    sign = -1 if text.startswith("-") else 1
    body = text.lstrip("+-").replace("UTC", "")
    if body.replace(":", "").replace(".", "").isdigit():
        hours, _, minutes = body.partition(":")
        return timezone(sign * timedelta(hours=float(hours), minutes=int(minutes or 0)))
    if ZoneInfo is None:
        raise ValueError("zoneinfo non disponibile")
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Fuso orario sconosciuto: {text}") from exc


def _julian_day(dt_utc: datetime) -> float:
    return dt_utc.timestamp() / 86400.0 + 2440587.5


def _refraction(elevation: float) -> float:
    if elevation > 85:
        return 0.0
    te = math.tan(math.radians(elevation))
    if elevation > 5:
        r = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
    elif elevation > -0.575:
        r = 1735 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)))
    else:
        r = -20.772 / te
    return r / 3600


def sun_position(when: datetime, lat: float, lon: float) -> SunPosition:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    utc = when.astimezone(timezone.utc)
    jc = (_julian_day(utc) - 2451545.0) / 36525.0

    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    center = (
        math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * mr) * 0.000289
    )
    omega = math.radians(125.04 - 1934.136 * jc)
    app_long = l0 + center - 0.00569 - 0.00478 * math.sin(omega)
    mean_obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq = math.radians(mean_obliq + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(obliq) * math.sin(math.radians(app_long)))

    y = math.tan(obliq / 2) ** 2
    l0r = math.radians(l0)
    eq_time = 4 * math.degrees(
        y * math.sin(2 * l0r)
        - 2 * ecc * math.sin(mr)
        + 4 * ecc * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r)
        - 1.25 * ecc * ecc * math.sin(2 * mr)
    )

    minutes = utc.hour * 60 + utc.minute + utc.second / 60 + utc.microsecond / 6e7
    true_solar = (minutes + eq_time + 4 * lon) % 1440
    hour_angle = true_solar / 4 - 180 if true_solar >= 0 else true_solar / 4 + 180

    latr, har = math.radians(lat), math.radians(hour_angle)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(har)
    zenith = math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))
    elevation = 90 - zenith

    denom = math.cos(latr) * math.sin(math.radians(zenith))
    if abs(denom) < 1e-9:
        azimuth = 180.0 if lat > 0 else 0.0
    else:
        cos_az = max(-1.0, min(1.0, (math.sin(latr) * math.cos(math.radians(zenith)) - math.sin(decl)) / denom))
        acos_az = math.degrees(math.acos(cos_az))
        azimuth = (acos_az + 180) % 360 if hour_angle > 0 else (540 - acos_az) % 360

    return SunPosition(when=when, azimuth=round(azimuth, 3), elevation=round(elevation + _refraction(elevation), 3))


def _day_samples(day: date, lat: float, lon: float, tz: tzinfo, step_min: int = 1) -> list[SunPosition]:
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    return [sun_position(start + timedelta(minutes=i), lat, lon) for i in range(0, 24 * 60, step_min)]


def _crossing(samples: list[SunPosition], threshold: float, rising: bool) -> datetime | None:
    for a, b in zip(samples, samples[1:]):
        if rising and a.elevation < threshold <= b.elevation:
            return b.when
        if not rising and a.elevation >= threshold > b.elevation:
            return b.when
    return None


def sun_times(day: date, lat: float, lon: float, tz: tzinfo) -> dict[str, datetime | None]:
    samples = _day_samples(day, lat, lon, tz)
    noon = max(samples, key=lambda s: s.elevation)
    return {
        "alba": _crossing(samples, -0.833, rising=True),
        "fine ora d'oro (mattina)": _crossing(samples, 6.0, rising=True),
        "mezzogiorno solare": noon.when,
        "inizio ora d'oro (sera)": _crossing(samples, 6.0, rising=False),
        "tramonto": _crossing(samples, -0.833, rising=False),
        "elevazione massima": noon.elevation,  # type: ignore[dict-item]
    }


@dataclass(frozen=True)
class ShadowMatch:
    when: datetime
    elevation: float
    azimuth: float
    shadow_azimuth: float
    error_deg: float


def times_for_shadow(
    day: date,
    lat: float,
    lon: float,
    tz: tzinfo,
    object_height: float,
    shadow_length: float,
    shadow_azimuth: float | None = None,
    tolerance_deg: float = 1.5,
) -> list[ShadowMatch]:
    """Times of the day whose sun elevation (and direction) match a shadow."""
    if object_height <= 0 or shadow_length <= 0:
        raise ValueError("Altezza e lunghezza dell'ombra devono essere > 0")
    target = math.degrees(math.atan(object_height / shadow_length))
    samples = [s for s in _day_samples(day, lat, lon, tz) if s.elevation > 0]

    def error(s: SunPosition) -> float:
        err = abs(s.elevation - target)
        if shadow_azimuth is not None:
            diff = abs((s.shadow_azimuth - shadow_azimuth + 180) % 360 - 180)
            err = math.hypot(err, diff / 4)  # direction weighs less: it is harder to measure
        return err

    # Local minima of the error function within tolerance (typically one morning, one afternoon).
    matches: list[ShadowMatch] = []
    for prev, cur, nxt in zip(samples, samples[1:], samples[2:]):
        e = error(cur)
        if e <= tolerance_deg and e <= error(prev) and e < error(nxt):
            matches.append(ShadowMatch(cur.when, cur.elevation, cur.azimuth, cur.shadow_azimuth, round(e, 3)))
    return sorted(matches, key=lambda m: m.error_deg)
