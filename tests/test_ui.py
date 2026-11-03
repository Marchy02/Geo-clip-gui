import pytest

pytest.importorskip("gradio")
pytest.importorskip("folium")

from geoosint.ui import handlers as h  # noqa: E402


def test_coordinate_tool():
    card, map_html, links = h.coordinate_tool("45.4408, 12.3155", False)
    assert "45.440800" in card and "<iframe" in map_html and "Street View" in links
    card, _, _ = h.coordinate_tool("boh", False)
    assert "non riconosciuto" in card


def test_sun_and_shadow_tools():
    card, svg = h.sun_tool("41.89, 12.49", "2024-06-21", "16:30", "Europe/Rome")
    assert "Azimut" in card and "<svg" in svg
    out = h.shadow_tool("41.89, 12.49", "2024-06-21", "Europe/Rome", 1.8, 2.0, 0)
    assert "16:4" in out and "09:3" in out


def test_ocr_language_rules():
    codes, note = h.normalize_ocr_languages(["Italiano", "Russo", "Giapponese"])
    assert codes == ["en", "ru"] and note
    assert h.normalize_ocr_languages(["Italiano", "Tedesco"]) == (["it", "de"], "")
    assert h.normalize_ocr_languages([]) == (["en"], "")


def test_run_analysis_outputs(gps_jpeg):
    out = h.run_analysis(str(gps_jpeg), ["Metadati", "Forense"], "world", 5, ["Italiano"], "", "", False, "")
    assert len(out) == 17
    verdict, map_html = out[0], out[1]
    assert "41.900000" in verdict and "<iframe" in map_html
    assert any(row[0] == "BodySerialNumber" for row in out[6])
    assert out[7] is not None and out[14]  # ELA image, report files
    empty = h.run_analysis(None, [], "world", 5, [], "", "", False, "")
    assert len(empty) == 17


def test_privacy_and_compare(gps_jpeg):
    html, clean = h.privacy_tool(str(gps_jpeg))
    assert clean and "presente → assente" in html
    cmp = h.compare_tool(str(gps_jpeg), clean)
    assert "stessa immagine" in cmp.lower() or "identici" in cmp


def test_batch(gps_jpeg, plain_png):
    rows, map_html, files = h.batch_tool([str(gps_jpeg), str(plain_png)], ["Metadati"], "world")
    assert len(rows) == 2 and rows[0][3].startswith("GPS") and len(files) == 2


def test_build_app():
    from geoosint.ui.app import build_app

    assert build_app() is not None
