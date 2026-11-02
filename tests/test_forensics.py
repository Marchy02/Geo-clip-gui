from PIL import Image, PngImagePlugin

from geoosint.forensics import compute_hashes, extract_metadata, hamming, parse_xmp, strip_metadata
from geoosint.forensics.analysis import run_forensics
from geoosint.forensics.jpeg import estimate_jpeg_quality
from geoosint.forensics.metadata import dms_to_decimal


def test_dms_to_decimal():
    assert dms_to_decimal((41.0, 54.0, 0.0), "N") == 41.9
    assert dms_to_decimal(((12, 1), (30, 1), (3600, 100)), "W") == -12.51
    assert dms_to_decimal((10.0, 0.0, 0.0), b"S") == -10.0


def test_extract_metadata_gps_and_findings(gps_jpeg):
    m = extract_metadata(gps_jpeg)
    assert m.camera == "Canon EOS 5D"
    assert m.gps and m.gps.lat == 41.9 and m.gps.lon == -12.51
    assert m.gps.altitude_m == 25.3
    assert m.datetime_original == "2024:06:21 16:47:00"
    assert m.exif["BodySerialNumber"] == "SN123"
    text = " ".join(m.findings)
    assert "GPS" in text and "Photoshop" in text and "serie" in text and "modific" in text
    assert m.jpeg_quality in range(70, 81)


def test_png_without_metadata(plain_png):
    m = extract_metadata(plain_png)
    assert m.gps is None and m.exif == {}
    assert m.jpeg_quality is None
    assert any("ripulita" in f for f in m.findings)


def test_ai_markers_in_png(tmp_path, scene):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a photo of Rome, Steps: 30, Sampler: Euler a")
    path = tmp_path / "sd.png"
    scene().save(path, pnginfo=info)
    m = extract_metadata(path)
    assert any("Stable Diffusion" in x for x in m.ai_markers)


def test_parse_xmp():
    xmp = b'<x:xmpmeta><rdf:Description xmp:CreatorTool="Adobe Photoshop" Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"><dc:title>Ciao</dc:title></rdf:Description></x:xmpmeta>'
    out = parse_xmp(xmp)
    assert out["xmp:CreatorTool"] == "Adobe Photoshop"
    assert out["dc:title"] == "Ciao"
    assert "trainedAlgorithmicMedia" in out["Iptc4xmpExt:DigitalSourceType"]


def test_jpeg_quality_estimate(tmp_path, scene):
    for q in (50, 75, 90, 95):
        path = tmp_path / f"q{q}.jpg"
        scene().save(path, quality=q)
        with Image.open(path) as img:
            assert abs(estimate_jpeg_quality(img) - q) <= 2


def test_hashes_survive_resize(tmp_path, scene):
    a, b, c = tmp_path / "a.png", tmp_path / "b.jpg", tmp_path / "c.png"
    scene().save(a)
    scene().resize((320, 240)).save(b, quality=70)
    Image.new("RGB", (640, 480), "black").save(c)
    ha, hb, hc = compute_hashes(a), compute_hashes(b), compute_hashes(c)
    assert ha.sha256 != hb.sha256 and len(ha.sha256) == 64
    assert hamming(ha.phash, hb.phash) <= 6
    assert hamming(ha.dhash, hb.dhash) <= 6
    assert hamming(ha.phash, hc.phash) > 10


def test_run_forensics(gps_jpeg):
    r = run_forensics(gps_jpeg)
    assert r.hashes and r.ela_score is not None
    assert r.ela_image.size == r.noise_image.size
    assert r.findings


def test_strip_metadata(gps_jpeg, tmp_path):
    out = strip_metadata(gps_jpeg, tmp_path / "clean.jpg")
    m = extract_metadata(out)
    assert m.exif == {} and m.gps is None and m.xmp == {}
    with Image.open(out) as img:
        assert img.size == (640, 480)
