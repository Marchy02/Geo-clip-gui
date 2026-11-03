import json

import pytest

from geoosint.models import AiVerdict, AnalysisReport, GeoPoint, GpsInfo, LocationSource, MetadataReport
from geoosint.pipeline import AnalysisOptions, Pipeline, fuse, validate_image
from geoosint.report import save_reports, to_html, to_markdown

NO_ML = AnalysisOptions(geoclip=False, ocr=False, yolo=False, ai=False)


def test_pipeline_without_ml(gps_jpeg):
    seen = []
    r = Pipeline().run(gps_jpeg, NO_ML, progress=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0
    assert [s.name for s in r.steps] == ["Metadati", "Forense"]
    assert all(s.status == "ok" for s in r.steps)
    assert r.final.source is LocationSource.EXIF and r.final.confidence == "alta"


def test_pipeline_isolates_failures(gps_jpeg, monkeypatch):
    from geoosint.engines import geoclip_engine
    from geoosint.engines.base import EngineUnavailable

    def boom(*a, **k):
        raise EngineUnavailable("GeoCLIP: test")

    monkeypatch.setattr(geoclip_engine, "predict", boom)
    r = Pipeline().run(gps_jpeg, AnalysisOptions(ocr=False, yolo=False, ai=False))
    status = {s.name: s.status for s in r.steps}
    assert status["GeoCLIP"] == "skipped" and status["Metadati"] == "ok"
    assert r.final is not None


def test_ai_step_skipped_without_key(gps_jpeg):
    r = Pipeline().run(gps_jpeg, AnalysisOptions(geoclip=False, ocr=False, yolo=False, ai=True))
    assert {s.name: s.status for s in r.steps}["Arbitro IA"] == "skipped"


def test_validate_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_image(tmp_path / "nope.jpg")
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image")
    with pytest.raises(ValueError):
        validate_image(fake)


def _report(**kw):
    r = AnalysisReport(image_path="x.jpg")
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_fuse_priorities():
    geo = [GeoPoint(10, 10, 0.5), GeoPoint(11, 11, 0.2)]
    ai = AiVerdict(override=True, lat=45.0, lon=9.0, confidence=80, location_name="Milano")
    gps = MetadataReport(gps=GpsInfo(45.01, 9.01))
    assert fuse(_report(geoclip=geo)).source is LocationSource.GEOCLIP
    assert fuse(_report(geoclip=geo, ai=ai)).source is LocationSource.AI
    final = fuse(_report(geoclip=geo, ai=ai, metadata=gps))
    assert final.source is LocationSource.EXIF and "concordante" in final.explanation
    far = fuse(_report(ai=ai, metadata=MetadataReport(gps=GpsInfo(-33.0, 151.0))))
    assert "DISCORDANZA" in far.explanation
    weak = fuse(_report(ai=AiVerdict(override=False, lat=1.0, lon=2.0, confidence=20)))
    assert weak.source is LocationSource.AI and weak.confidence == "bassa"
    assert fuse(_report()) is None


def test_reports_escape_untrusted_text(gps_jpeg, tmp_path):
    from geoosint.models import OcrText

    r = Pipeline().run(gps_jpeg, NO_ML)
    r.ocr = [OcrText("<script>alert(1)</script>", 0.9)]
    r.ai = AiVerdict(override=True, lat=1.0, lon=2.0, location_name="<img src=x onerror=alert(1)>", reasoning="ok")
    html = to_html(r)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html
    assert "# Report OSINT" in to_markdown(r)
    paths = save_reports(r, tmp_path / "out")
    assert set(paths) == {"json", "md", "html"}
    data = json.loads(paths["json"].read_text())
    assert data["final"]["source"] == "exif" and "ela_image" not in data["forensics"]
