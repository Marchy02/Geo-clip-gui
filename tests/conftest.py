import os
from fractions import Fraction

import pytest
from PIL import ExifTags, Image, ImageDraw

# Tests must never hit the network.
os.environ["ONLINE_LOOKUPS"] = "false"
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)


@pytest.fixture(autouse=True)
def _isolated_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    from geoosint.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _scene(size=(640, 480)) -> Image.Image:
    img = Image.new("RGB", size, "#87ceeb")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, size[1] * 0.6, size[0], size[1]), fill="#556b2f")
    draw.rectangle((80, 150, 300, 330), fill="#a52a2a")
    draw.ellipse((420, 60, 500, 140), fill="#ffd700")
    return img


@pytest.fixture
def gps_jpeg(tmp_path):
    img = _scene()
    exif = img.getexif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "Canon EOS 5D"
    exif[0x0131] = "Adobe Photoshop 25.0"
    exif[0x0132] = "2024:06:22 10:00:00"
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    sub[0x9003] = "2024:06:21 16:47:00"
    sub[0xA431] = "SN123"
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps[1], gps[2] = "N", (41.0, 54.0, 0.0)
    gps[3], gps[4] = "W", (12.0, 30.0, 36.0)
    gps[5], gps[6] = 0, Fraction(253, 10)
    path = tmp_path / "gps.jpg"
    img.save(path, exif=exif, quality=75)
    return path


@pytest.fixture
def plain_png(tmp_path):
    path = tmp_path / "plain.png"
    _scene().save(path)
    return path


@pytest.fixture
def scene():
    return _scene
