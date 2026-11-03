import json

from geoosint.cli import main


def test_cli_coords_json(capsys):
    assert main(["coords", "45.4408", "12.3155", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["formats"]["Plus Code"].startswith("8FQJ")


def test_cli_bad_coords():
    assert main(["coords", "boh"]) == 2


def test_cli_exif_and_strip(gps_jpeg, tmp_path, capsys):
    assert main(["exif", str(gps_jpeg), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["gps"]["lat"] == 41.9
    out = tmp_path / "c.jpg"
    assert main(["strip", str(gps_jpeg), "-o", str(out)]) == 0 and out.exists()
    capsys.readouterr()
    assert main(["compare", str(gps_jpeg), str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["perceptual_distance"] <= 6


def test_cli_analyze_writes_reports(gps_jpeg, tmp_path):
    rc = main(
        ["analyze", str(gps_jpeg), "--no-geoclip", "--no-ocr", "--no-yolo", "--no-ai", "-q", "-o", str(tmp_path), "--format", "json", "md"]
    )
    assert rc == 0
    assert len(list(tmp_path.glob("*.json"))) == 1 and len(list(tmp_path.glob("*.md"))) == 1


def test_cli_sun_and_shadow(capsys):
    assert main(["sun", "41.89", "12.49", "--date", "2024-06-21", "--tz", "Europe/Rome"]) == 0
    assert "Alba" in capsys.readouterr().out
    assert main(["shadow", "41.89", "12.49", "--date", "2024-06-21", "--tz", "Europe/Rome", "--height", "1.8", "--length", "2"]) == 0
