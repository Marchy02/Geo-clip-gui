from datetime import date, datetime, timedelta, timezone

import pytest

from geoosint.geo.sun import parse_tz, sun_position, sun_times, times_for_shadow

ROME = (41.9028, 12.4964)


def test_parse_tz():
    assert parse_tz("UTC") is timezone.utc
    assert parse_tz("+02:00").utcoffset(None) == timedelta(hours=2)
    assert parse_tz("-5").utcoffset(None) == timedelta(hours=-5)
    assert str(parse_tz("Europe/Rome")) == "Europe/Rome"
    with pytest.raises(ValueError):
        parse_tz("Mars/Olympus")


def test_summer_solstice_noon_elevation():
    # Max elevation at Rome on the June solstice ≈ 90 - 41.9 + 23.44 = 71.5°
    times = sun_times(date(2024, 6, 21), *ROME, parse_tz("Europe/Rome"))
    assert times["max_elevation"] == pytest.approx(71.5, abs=0.3)
    assert times["sunrise"].strftime("%H:%M") in {"05:34", "05:35"}
    assert times["sunset"].strftime("%H:%M") in {"20:48", "20:49"}


def test_sun_position_reference():
    # NOAA calculator: Rome 2024-06-21 12:00 UTC -> az ~ 212°, el ~ 69°
    pos = sun_position(datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc), 41.9, 12.5)
    assert pos.azimuth == pytest.approx(212.2, abs=0.5)
    assert pos.elevation == pytest.approx(69.0, abs=0.3)
    assert pos.shadow_azimuth == pytest.approx(32.2, abs=0.5)


def test_southern_hemisphere_sun_is_north_at_noon():
    tz = parse_tz("Australia/Sydney")
    noon = sun_times(date(2024, 6, 21), -33.87, 151.21, tz)["solar_noon"]
    assert abs(sun_position(noon, -33.87, 151.21).azimuth % 360) < 5 or sun_position(noon, -33.87, 151.21).azimuth > 355


def test_shadow_finds_both_solutions_and_direction_disambiguates():
    tz = parse_tz("Europe/Rome")
    both = times_for_shadow(date(2024, 6, 21), *ROME, tz, 1.8, 2.0)
    assert len(both) == 2
    hours = sorted(m.when.hour for m in both)
    assert hours[0] < 12 < hours[1]
    afternoon = both[0] if both[0].when.hour > 12 else both[1]
    one = times_for_shadow(date(2024, 6, 21), *ROME, tz, 1.8, 2.0, shadow_azimuth=afternoon.shadow_azimuth)
    assert one[0].when.hour == afternoon.when.hour


def test_shadow_impossible_and_invalid():
    tz = parse_tz("Europe/Rome")
    assert times_for_shadow(date(2024, 12, 21), *ROME, tz, 10, 1) == []
    with pytest.raises(ValueError):
        times_for_shadow(date(2024, 12, 21), *ROME, tz, 0, 1)
