import math

import pytest

from geoosint.geo.coords import (
    CoordinateError,
    all_formats,
    from_utm,
    geohash_decode,
    geohash_encode,
    haversine_km,
    initial_bearing,
    parse_coordinates,
    plus_code_decode,
    plus_code_encode,
    to_dms,
    to_utm,
    weighted_centroid,
)


@pytest.mark.parametrize(
    ("text", "lat", "lon"),
    [
        ("45.4408, 12.3155", 45.4408, 12.3155),
        ("45,4408; 12,3155", 45.4408, 12.3155),
        ("-33.86 151.2", -33.86, 151.2),
        ("45°26'27\"N 12°18'56\"E", 45.440833, 12.315556),
        ("40°41'21.4\"N 74°02'40.2\"W", 40.689278, -74.0445),
        ("N 45 26.45 E 12 18.93", 45.440833, 12.3155),
        ("33°51′S, 151°12′E", -33.85, 151.2),
        ("45.44N 12.31O", 45.44, -12.31),
        ("https://www.google.com/maps/place/X/@45.43,12.33,17z/data=!3d45.4408!4d12.3155", 45.4408, 12.3155),
        ("https://www.google.com/maps/@45.43,12.33,17z", 45.43, 12.33),
        ("https://maps.google.com/?q=45.1,9.2", 45.1, 9.2),
        ("https://www.openstreetmap.org/#map=17/45.44/12.31", 45.44, 12.31),
        ("https://www.bing.com/maps?cp=45.5~9.1&lvl=11", 45.5, 9.1),
        ("geo:45.1,9.2", 45.1, 9.2),
        ("8FVC9G8F+6X", 47.365563, 8.524938),
        ("u4pruydqqvj", 57.649111, 10.40744),
    ],
)
def test_parse_coordinates(text, lat, lon):
    p = parse_coordinates(text)
    assert p.lat == pytest.approx(lat, abs=1e-4)
    assert p.lon == pytest.approx(lon, abs=1e-4)


@pytest.mark.parametrize("text", ["", "foo", "95, 10", "45, 200", "N 45 E"])
def test_parse_coordinates_rejects(text):
    with pytest.raises(CoordinateError):
        parse_coordinates(text)


def test_plus_code_reference():
    assert plus_code_encode(47.0000625, 8.0000625) == "8FVC2222+22"
    p = plus_code_decode(plus_code_encode(-33.8688, 151.2093))
    assert haversine_km(p.lat, p.lon, -33.8688, 151.2093) < 0.02


def test_geohash_roundtrip():
    assert geohash_encode(57.64911, 10.40744, 11) == "u4pruydqqvj"
    p = geohash_decode(geohash_encode(-22.9, -43.2, 10))
    assert (p.lat, p.lon) == pytest.approx((-22.9, -43.2), abs=1e-5)


@pytest.mark.parametrize(("lat", "lon"), [(45.4408, 12.3155), (-33.8688, 151.2093), (40.6892, -74.0445), (64.1, -21.9), (0.5, 0.5)])
def test_utm_roundtrip(lat, lon):
    zone_band, e, n = to_utm(lat, lon).split()
    back = from_utm(int(zone_band[:-1]), zone_band[-1], float(e), float(n))
    assert haversine_km(lat, lon, back.lat, back.lon) < 0.002


def test_utm_reference_value():
    # Statue of Liberty: 18T 580735 4504695 (±10 m)
    zone, e, n = to_utm(40.689247, -74.044502).split()
    assert zone == "18T"
    assert abs(float(e) - 580735) < 15 and abs(float(n) - 4504695) < 15


def test_dms_format():
    assert to_dms(45.440833, -12.315556) == "45°26'27.0\"N 12°18'56.0\"W"


def test_distance_and_bearing():
    # Rome -> Milan ≈ 477 km, roughly north-west
    d = haversine_km(41.9028, 12.4964, 45.4642, 9.19)
    assert 470 < d < 485
    assert 300 < initial_bearing(41.9028, 12.4964, 45.4642, 9.19) < 340


def test_weighted_centroid():
    lat, lon, spread = weighted_centroid([(45, 12, 0.5), (41, 12, 0.5)])
    assert lat == pytest.approx(43, abs=0.01) and lon == pytest.approx(12, abs=0.01)
    assert spread == pytest.approx(haversine_km(43, 12, 45, 12), rel=0.01)
    lat, _, _ = weighted_centroid([(45, 12, 0.9), (41, 12, 0.1)])
    assert lat > 44


def test_all_formats_keys():
    f = all_formats(45.0, 9.0)
    assert set(f) == {"Decimale", "DMS", "DDM", "UTM (WGS84)", "Plus Code", "Geohash"}
    assert not any(math.isnan(float(x)) for x in f["Decimale"].split(", "))
