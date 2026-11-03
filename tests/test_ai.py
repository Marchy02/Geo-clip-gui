import pytest

from geoosint.ai.common import build_context, parse_verdict, verdict_from_dict
from geoosint.models import Detection, GeoPoint, OcrText


def test_parse_verdict_variants():
    assert parse_verdict('{"override": true}') == {"override": True}
    assert parse_verdict('```json\n{"lat": 1}\n```') == {"lat": 1}
    assert parse_verdict('Ecco il risultato:\n{"lat": 2, "lon": 3}\nSpero sia utile') == {"lat": 2, "lon": 3}
    with pytest.raises(ValueError):
        parse_verdict("nessun json qui")
    with pytest.raises(ValueError):
        parse_verdict("")


def test_verdict_from_dict_validation():
    v = verdict_from_dict({"override": True, "lat": "45.1", "lon": 12, "confidence": 140, "clues": "insegna in italiano"})
    assert v.override and v.lat == 45.1 and v.confidence == 100 and v.clues == ["insegna in italiano"]
    bad = verdict_from_dict({"override": True, "lat": 123, "lon": 12})
    assert bad.lat is None and not bad.override and not bad.has_location
    legacy = verdict_from_dict({"override": False, "description": "vecchio formato"})
    assert legacy.reasoning == "vecchio formato"


def test_build_context_contains_all_clues():
    ctx = build_context(
        [GeoPoint(45.0, 9.0, 0.3, address="Milano")],
        [OcrText("Apotheke", 0.9)],
        [Detection("car", 0.8), Detection("car", 0.7)],
        None,
        "foto di febbraio",
    )
    assert "Milano" in ctx and "Apotheke" in ctx and "car×2" in ctx and "febbraio" in ctx
