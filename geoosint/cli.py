"""Command line interface: every tool of the dashboard, scriptable.

Examples:
    geoosint analyze foto.jpg --no-ai --format html json
    geoosint batch ./cartella --output reports/ --no-ocr
    geoosint exif foto.jpg --json
    geoosint coords "45°26'27\"N 12°18'56\"E" --reverse
    geoosint sun "41.89, 12.49" --date 2024-06-21 --time 16:30 --tz Europe/Rome
    geoosint shadow "41.89, 12.49" --date 2024-06-21 --height 1.8 --length 2.0
    geoosint strip foto.jpg -o foto_pulita.jpg
    geoosint compare a.jpg b.jpg
    geoosint ui --port 7860
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from . import __version__
from .config import get_settings
from .models import to_jsonable


def _print_json(obj) -> None:
    print(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2))


def _rows(rows: list[tuple[str, object]]) -> None:
    rows = [(k, v) for k, v in rows if v not in (None, "")]
    width = max((len(k) for k, _ in rows), default=0)
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")


def _options_from_args(args):
    from .pipeline import AnalysisOptions

    return AnalysisOptions(
        metadata=not args.no_metadata,
        forensics=not args.no_forensics,
        geoclip=not args.no_geoclip,
        ocr=not args.no_ocr,
        yolo=not args.no_yolo,
        ai=not args.no_ai,
        reverse_geocode=not args.offline,
        geoclip_preset=args.preset,
        top_k=args.top_k,
        ocr_languages=args.ocr_lang,
        ai_provider=args.provider,
        ai_model=args.model,
        analyst_notes=args.notes or "",
    )


def _summary(report) -> None:
    print(f"\n{Path(report.image_path).name}")
    if report.final:
        f = report.final
        print(f"  Verdetto     {f.point.lat:.6f}, {f.point.lon:.6f}  [{f.source.label} · affidabilità {f.confidence}]")
        if f.point.label:
            print(f"  Luogo        {f.point.label}")
        if f.point.address:
            print(f"  Indirizzo    {f.point.address}")
        print(f"  Motivazione  {f.explanation}")
    else:
        print("  Nessuna posizione determinata.")
    for i, g in enumerate(report.geoclip[:5]):
        print(f"  GeoCLIP #{i + 1}   {g.lat:.5f}, {g.lon:.5f}  {(g.prob or 0) * 100:5.1f}%  {g.address}")
    if report.ocr:
        print("  OCR          " + " | ".join(t.text for t in report.ocr[:10]))
    if report.metadata:
        for item in [*report.metadata.ai_markers, *report.metadata.findings]:
            print(f"  - {item}")
    for s in report.steps:
        if s.status != "ok":
            print(f"  [{s.status}] {s.name}: {s.detail}")


def cmd_analyze(args) -> int:
    from .pipeline import Pipeline
    from .report import save_reports

    pipe = Pipeline()
    opts = _options_from_args(args)
    out = Path(args.output or get_settings().reports_dir)
    code = 0
    for image in args.images:
        try:
            report = pipe.run(image, opts, progress=None if args.quiet else lambda f, m: print(f"  … {m}", file=sys.stderr))
        except Exception as exc:
            print(f"Errore su {image}: {exc}", file=sys.stderr)
            code = 1
            continue
        if args.json:
            _print_json(report.to_dict())
        else:
            _summary(report)
        if args.format:
            for fmt, path in save_reports(report, out, tuple(args.format)).items():
                print(f"  report {fmt:4} → {path}", file=sys.stderr)
    return code


def cmd_batch(args) -> int:
    from .pipeline import SUPPORTED_EXTENSIONS

    folder = Path(args.folder)
    pattern = "**/*" if args.recursive else "*"
    args.images = sorted(str(p) for p in folder.glob(pattern) if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not args.images:
        print(f"Nessuna immagine in {folder}", file=sys.stderr)
        return 1
    print(f"{len(args.images)} immagini da analizzare", file=sys.stderr)
    return cmd_analyze(args)


def cmd_exif(args) -> int:
    from .forensics import extract_metadata

    meta = extract_metadata(args.image)
    if args.json:
        _print_json(meta)
        return 0
    print(f"{meta.file_name}  ({meta.format} {meta.width}×{meta.height}, {meta.file_size / 1024:.1f} KB)")
    _rows(
        [
            ("Fotocamera", meta.camera),
            ("Obiettivo", meta.lens),
            ("Software", meta.software),
            ("Data scatto", meta.datetime_original),
            ("Fuso", meta.timezone_offset),
            ("GPS", f"{meta.gps.lat:.6f}, {meta.gps.lon:.6f}" if meta.gps else None),
            ("Altitudine", f"{meta.gps.altitude_m} m" if meta.gps and meta.gps.altitude_m is not None else None),
            ("Qualità JPEG", meta.jpeg_quality),
        ]
    )
    if args.all:
        print("\n  -- EXIF --")
        _rows(list(meta.exif.items()))
        if meta.xmp:
            print("\n  -- XMP --")
            _rows(list(meta.xmp.items()))
    for item in [*meta.ai_markers, *meta.findings]:
        print(f"  - {item}")
    return 0


def cmd_hash(args) -> int:
    from .forensics import compute_hashes

    for image in args.images:
        h = compute_hashes(image)
        if args.json:
            _print_json({"file": image, **to_jsonable(h)})
        else:
            print(image)
            _rows([("SHA-256", h.sha256), ("SHA-1", h.sha1), ("MD5", h.md5), ("pHash", h.phash), ("dHash", h.dhash), ("aHash", h.ahash)])
    return 0


def cmd_strip(args) -> int:
    from .forensics import strip_metadata

    src = Path(args.image)
    dst = Path(args.output) if args.output else src.with_name(f"{src.stem}_pulita{src.suffix}")
    print(strip_metadata(src, dst))
    return 0


def cmd_compare(args) -> int:
    from .forensics import compute_hashes, hamming

    a, b = compute_hashes(args.a), compute_hashes(args.b)
    dist = min(hamming(a.phash, b.phash), hamming(a.dhash, b.dhash))
    same = a.sha256 == b.sha256
    verdict = (
        "identiche"
        if same
        else "stessa immagine (ricompressa/ridimensionata)"
        if dist <= 6
        else "molto simili"
        if dist <= 14
        else "diverse"
    )
    if args.json:
        _print_json({"identical": same, "perceptual_distance": dist, "verdict": verdict})
    else:
        print(f"Distanza percettiva: {dist}/64 → {verdict}")
    return 0


def cmd_coords(args) -> int:
    from .geo import geocode
    from .geo.coords import CoordinateError, all_formats, parse_coordinates
    from .geo.links import map_links

    try:
        p = parse_coordinates(" ".join(args.text))
    except CoordinateError as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2
    data: dict = {"lat": p.lat, "lon": p.lon, "formats": all_formats(p.lat, p.lon)}
    if args.reverse:
        try:
            data["address"] = geocode.reverse_geocode(p.lat, p.lon, zoom=18)
        except geocode.GeocodingError as exc:
            data["address_error"] = str(exc)
    if args.links:
        data["links"] = {link.name: link.url for link in map_links(p.lat, p.lon)}
    if args.json:
        _print_json(data)
        return 0
    _rows(list(data["formats"].items()))
    if "address" in data:
        _rows([("Indirizzo", data["address"].get("display_name"))])
    for name, url in data.get("links", {}).items():
        print(f"  {name:26} {url}")
    return 0


def _parse_when(args, tz):
    day = date.fromisoformat(args.date) if args.date else date.today()
    hh, mm = (int(x) for x in (args.time or "12:00").split(":")[:2])
    return day, datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)


def cmd_sun(args) -> int:
    from .geo.coords import compass_point, parse_coordinates
    from .geo.sun import SUN_TIME_LABELS, parse_tz, sun_position, sun_times

    p = parse_coordinates(" ".join(args.place))
    tz = parse_tz(args.tz)
    day, when = _parse_when(args, tz)
    pos = sun_position(when, p.lat, p.lon)
    times = sun_times(day, p.lat, p.lon, tz)
    if args.json:
        _print_json({"position": pos, "times": times})
        return 0
    print(f"{p} · {when:%Y-%m-%d %H:%M %Z}")
    _rows(
        [
            ("Azimut", f"{pos.azimuth:.1f}° ({compass_point(pos.azimuth)})"),
            ("Elevazione", f"{pos.elevation:.2f}°"),
            ("Ombre verso", f"{pos.shadow_azimuth:.1f}° ({compass_point(pos.shadow_azimuth)})"),
            ("Ombra/altezza", f"{pos.shadow_ratio:.2f}" if pos.shadow_ratio else "sole sotto l'orizzonte"),
            *[
                (SUN_TIME_LABELS[k], v.strftime("%H:%M") if hasattr(v, "strftime") else f"{v:.1f}°")
                for k, v in times.items()
                if v is not None
            ],
        ]
    )
    return 0


def cmd_shadow(args) -> int:
    from .geo.coords import parse_coordinates
    from .geo.sun import parse_tz, times_for_shadow

    p = parse_coordinates(" ".join(args.place))
    tz = parse_tz(args.tz)
    day = date.fromisoformat(args.date) if args.date else date.today()
    matches = times_for_shadow(day, p.lat, p.lon, tz, args.height, args.length, args.direction)
    if args.json:
        _print_json(matches)
        return 0
    if not matches:
        print("Nessun orario compatibile.")
        return 1
    for m in matches:
        print(
            f"  {m.when:%H:%M}  elev {m.elevation:5.1f}°  sole {m.azimuth:5.1f}°  ombra {m.shadow_azimuth:5.1f}°  errore ±{m.error_deg:.2f}°"
        )
    return 0


def cmd_ui(args) -> int:
    from .ui.app import main as ui_main

    argv = (
        ["--host", args.host, "--port", str(args.port)] + (["--share"] if args.share else []) + (["--auth", args.auth] if args.auth else [])
    )
    ui_main(argv + (["--inbrowser"] if args.inbrowser else []))
    return 0


def cmd_status(args) -> int:
    from .engines import statuses

    s = get_settings()
    for st in statuses():
        print(f"  {st.name:8} {'OK ' if st.installed else 'NO '} {st.detail}")
    active = s.provider()
    for p in s.ai_providers:
        mark = "*" if active and p.id == active.id else " "
        state = "OK " if p.is_ready() else "NO "
        print(
            f" {mark}IA       {state} {p.id} ({p.kind}/{p.model or '?'}) chiave {p.key_source()}"
            + (f" — {p.problem()}" if p.problem() else "")
        )
    from .user_config import config_path

    print(f"  Impostazioni salvate in {config_path()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    s = get_settings()
    parser = argparse.ArgumentParser(prog="geoosint", description="GeoOSINT — coltellino svizzero OSINT per immagini")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log dettagliati")
    sub = parser.add_subparsers(dest="command", required=True, metavar="comando")

    def analysis_flags(p):
        p.add_argument("--no-metadata", action="store_true")
        p.add_argument("--no-forensics", action="store_true")
        p.add_argument("--no-geoclip", action="store_true")
        p.add_argument("--no-ocr", action="store_true")
        p.add_argument("--no-yolo", action="store_true")
        p.add_argument("--no-ai", action="store_true", help="non usare l'arbitro IA")
        p.add_argument("--offline", action="store_true", help="niente reverse geocoding")
        p.add_argument("--preset", choices=["world", "ukraine_russia"], default=s.geoclip_preset)
        p.add_argument("--top-k", type=int, default=s.geoclip_top_k)
        p.add_argument("--ocr-lang", nargs="+", metavar="LANG", help="codici EasyOCR (es. en it de)")
        p.add_argument("--provider", help="id del provider IA (vedi: geoosint status)")
        p.add_argument("--model", help="modello IA per questa analisi (sovrascrive quello del provider)")
        p.add_argument("--notes", help="contesto per l'arbitro IA")
        p.add_argument("--format", nargs="*", choices=["json", "md", "html"], default=["html", "json"], help="report da salvare")
        p.add_argument("-o", "--output", help="cartella dei report")
        p.add_argument("--json", action="store_true", help="stampa il risultato completo in JSON")
        p.add_argument("-q", "--quiet", action="store_true")

    p = sub.add_parser("analyze", help="analisi completa di una o più immagini")
    p.add_argument("images", nargs="+")
    analysis_flags(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("batch", help="analizza tutte le immagini di una cartella")
    p.add_argument("folder")
    p.add_argument("-r", "--recursive", action="store_true")
    analysis_flags(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("exif", help="metadati e osservazioni")
    p.add_argument("image")
    p.add_argument("--all", action="store_true", help="mostra tutti i tag")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_exif)

    p = sub.add_parser("hash", help="hash crittografici e percettivi")
    p.add_argument("images", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_hash)

    p = sub.add_parser("strip", help="crea una copia senza metadati")
    p.add_argument("image")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_strip)

    p = sub.add_parser("compare", help="confronta due immagini (hash percettivi)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("coords", help="converte coordinate in tutti i formati")
    p.add_argument("text", nargs="+")
    p.add_argument("--reverse", action="store_true", help="indirizzo da OpenStreetMap")
    p.add_argument("--links", action="store_true", help="link a mappe e servizi")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_coords)

    for name, func, helptext in (("sun", cmd_sun, "posizione del sole"), ("shadow", cmd_shadow, "orari compatibili con un'ombra")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("place", nargs="+", help="coordinate del luogo")
        p.add_argument("--date", help="AAAA-MM-GG (default oggi)")
        p.add_argument("--tz", default="UTC", help="fuso: Europe/Rome, +02:00, UTC")
        p.add_argument("--json", action="store_true")
        if name == "sun":
            p.add_argument("--time", default="12:00", help="HH:MM ora locale")
        else:
            p.add_argument("--height", type=float, required=True, help="altezza oggetto (m)")
            p.add_argument("--length", type=float, required=True, help="lunghezza ombra (m)")
            p.add_argument("--direction", type=float, help="direzione dell'ombra in gradi da Nord")
        p.set_defaults(func=func)

    p = sub.add_parser("ui", help="avvia la dashboard web")
    p.add_argument("--host", default=s.server_host)
    p.add_argument("--port", type=int, default=s.server_port)
    p.add_argument("--share", action="store_true")
    p.add_argument("--auth", help="utente:password")
    p.add_argument("--inbrowser", action="store_true")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("status", help="stato dei motori e della configurazione")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
