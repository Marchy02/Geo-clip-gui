"""Event handlers: pure functions from UI inputs to UI outputs (testable without a browser)."""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from datetime import date, datetime
from pathlib import Path

from .. import __version__
from ..config import get_settings
from ..engines import statuses
from ..engines.base import modules_available
from ..geo import geocode
from ..geo.coords import CoordinateError, all_formats, compass_point, haversine_km, initial_bearing, parse_coordinates
from ..geo.links import VERIFICATION_TOOLS, map_links, reverse_image_links
from ..geo.sun import day_samples, parse_tz, sun_position, sun_times, times_for_shadow
from ..models import AnalysisReport, StepStatus
from ..pipeline import AnalysisOptions, Pipeline
from ..report import save_reports
from . import components as ui
from .maps import Marker, empty_map, render_map

log = logging.getLogger(__name__)

MODULES = {
    "Metadati": "metadata",
    "Forense": "forensics",
    "GeoCLIP": "geoclip",
    "OCR": "ocr",
    "Oggetti (YOLO)": "yolo",
    "Arbitro IA (Gemini)": "ai",
    "Reverse geocoding": "reverse_geocode",
}
OCR_LANGS = {
    "Inglese": "en", "Italiano": "it", "Tedesco": "de", "Francese": "fr", "Spagnolo": "es", "Portoghese": "pt",
    "Olandese": "nl", "Polacco": "pl", "Ceco": "cs", "Rumeno": "ro", "Turco": "tr", "Russo": "ru", "Ucraino": "uk",
    "Bulgaro": "bg", "Serbo (cirillico)": "rs_cyrillic", "Arabo": "ar", "Persiano": "fa", "Hindi": "hi",
    "Cinese semplificato": "ch_sim", "Giapponese": "ja", "Coreano": "ko", "Thai": "th", "Vietnamita": "vi",
}  # fmt: skip
# EasyOCR: Latin languages combine freely; most other scripts must be paired with English only.
_SCRIPT_LANGS = {"ru", "uk", "bg", "rs_cyrillic", "ar", "fa", "hi", "ch_sim", "ja", "ko", "th"}


def header_html() -> str:
    s = get_settings()
    chips = []
    for st in statuses():
        chips.append((st.name, "ok" if st.loaded else ("" if st.installed else "bad")))
    chips.append(("Gemini", "ok" if s.gemini_api_key and modules_available("google.genai")[0] else "warn"))
    chips.append(("Online" if s.online_lookups else "Offline", "ok" if s.online_lookups else "warn"))
    return ui.header(__version__, chips)


def normalize_ocr_languages(labels: list[str]) -> tuple[list[str], str]:
    codes = [OCR_LANGS[label] for label in labels if label in OCR_LANGS] or ["en"]
    scripts = [c for c in codes if c in _SCRIPT_LANGS]
    if scripts:
        # Keep the first non-Latin script + English (EasyOCR compatibility rule).
        chosen = ["en", scripts[0]]
        note = "" if len(codes) == len(chosen) and set(codes) == set(chosen) else f"OCR: alfabeto non latino → uso solo {', '.join(chosen)}"
        return chosen, note
    return codes, ""


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _markers(report: AnalysisReport) -> list[Marker]:
    markers: list[Marker] = []
    if report.final:
        p = report.final.point
        markers.append(Marker(p.lat, p.lon, f"Verdetto: {p.label}", "final"))
    if report.metadata and report.metadata.gps:
        g = report.metadata.gps
        markers.append(Marker(g.lat, g.lon, "GPS EXIF", "exif"))
    if report.ai and report.ai.lat is not None and report.ai.lon is not None:
        markers.append(Marker(report.ai.lat, report.ai.lon, f"IA: {report.ai.location_name or 'ipotesi'} ({report.ai.confidence}%)", "ai"))
    for i, g in enumerate(report.geoclip):
        label = f"GeoCLIP #{i + 1} · {(g.prob or 0) * 100:.1f}%" + (f" · {g.address}" if g.address else "")
        markers.append(Marker(g.lat, g.lon, label, "geoclip", weight=g.prob))
    return markers


def run_analysis(
    image_path: str | None,
    modules: list[str],
    preset: str,
    top_k: int,
    ocr_labels: list[str],
    api_key: str,
    model: str,
    use_search: bool,
    notes: str,
    image_url: str = "",
    progress=None,
):
    """Return the tuple of outputs wired in ``app.py`` (see ``ANALYSIS_OUTPUTS``)."""
    if not image_path:
        return blank_outputs("Carica un'immagine (trascinala, incollala o scaricala da URL) e premi Analizza.")

    enabled = {MODULES[m] for m in modules if m in MODULES}
    langs, lang_note = normalize_ocr_languages(ocr_labels)
    opts = AnalysisOptions(
        metadata="metadata" in enabled,
        forensics="forensics" in enabled,
        geoclip="geoclip" in enabled,
        ocr="ocr" in enabled,
        yolo="yolo" in enabled,
        ai="ai" in enabled,
        reverse_geocode="reverse_geocode" in enabled,
        geoclip_preset=preset,
        top_k=int(top_k),
        ocr_languages=langs,
        gemini_api_key=(api_key or "").strip() or None,
        gemini_model=(model or "").strip() or None,
        gemini_search=bool(use_search),
        analyst_notes=notes or "",
    )

    def cb(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, desc=msg)

    try:
        report = Pipeline().run(image_path, opts, progress=cb)
    except Exception as exc:
        log.exception("Analisi fallita")
        return blank_outputs(f"Analisi impossibile: {exc}", error=True)

    if lang_note:
        report.steps.append(StepStatus(name="Lingue OCR", status="ok", detail=lang_note))

    report_files: list[str] = []
    try:
        paths = save_reports(report, get_settings().reports_dir)
        report_files = [str(p) for p in paths.values()]
    except Exception as exc:  # pragma: no cover - disk issues
        log.warning("Salvataggio report fallito: %s", exc)

    final = report.final
    if final:
        geo_html = ui.formats_card(final.point.lat, final.point.lon) + (
            f'<div class="card"><h3>🔗 Verifica sul posto</h3>{ui.links_html(map_links(final.point.lat, final.point.lon))}</div>'
        )
    else:
        geo_html = ui.empty("Nessuna coordinata da convertire.", "📐")

    geo_rows = [
        [i + 1, round(g.lat, 5), round(g.lon, 5), f"{(g.prob or 0) * 100:.2f}%", g.address or ""] for i, g in enumerate(report.geoclip)
    ]
    exif_rows = [[k, v] for k, v in (report.metadata.exif.items() if report.metadata else [])]
    exif_rows += [[f"XMP · {k}", v] for k, v in (report.metadata.xmp.items() if report.metadata else [])]
    exif_rows += [[f"PNG · {k}", v] for k, v in (report.metadata.png_text.items() if report.metadata else [])]

    from ..engines.yolo_engine import GEO_RELEVANT, summarize

    reverse = ui.links_html(reverse_image_links(image_url.strip() or None)) + ui.links_html(VERIFICATION_TOOLS)
    reverse_html = (
        '<div class="card"><h3>🔁 Ricerca inversa e verifica</h3>'
        + (
            ""
            if image_url.strip()
            else '<p class="muted">Immagine locale: apri un motore e carica il file. Con un URL pubblico i link cercano direttamente.</p>'
        )
        + reverse
        + "</div>"
    )

    fz = report.forensics
    return (
        ui.verdict_card(report),
        render_map(_markers(report)),
        geo_rows,
        geo_html,
        ui.ai_card(report.ai),
        ui.metadata_card(report.metadata),
        exif_rows,
        fz.ela_image if fz else None,
        fz.noise_image if fz else None,
        ui.forensics_card(fz),
        ui.ocr_card(report.ocr),
        report.detections_image,
        ui.objects_card(summarize(report.detections), GEO_RELEVANT),
        reverse_html,
        report_files or None,
        ui.steps_html(report.steps),
        report.to_dict(),
    )


def blank_outputs(message: str, error: bool = False):
    card = ui.error_card(message) if error else ui.empty(message)
    return (card, empty_map(), [], "", "", "", [], None, None, "", "", None, "", "", None, "", None)


def fetch_url(url: str):
    from ..fetch import FetchError, download_image

    if not (url or "").strip():
        return None, ui.error_card("Inserisci l'URL di un'immagine.")
    try:
        path = download_image(url)
    except FetchError as exc:
        return None, ui.error_card(str(exc))
    return str(path), f'<div class="card"><div class="finding good">✅ Scaricata: {ui._e(path.name)}</div></div>'


# ---------------------------------------------------------------------------
# Coordinates tab
# ---------------------------------------------------------------------------


def coordinate_tool(text: str, do_reverse: bool):
    try:
        point = parse_coordinates(text)
    except CoordinateError as exc:
        return ui.error_card(str(exc)), empty_map(380), ""
    info_html = ""
    if do_reverse and get_settings().online_lookups:
        try:
            info = geocode.reverse_geocode(point.lat, point.lon, zoom=18)
            info_html = ui.kv_table(
                [("Indirizzo", info.get("display_name")), ("Paese", f"{info.get('country', '')} {info.get('country_code', '')}".strip()),
                 ("Regione", info.get("state")), ("Città", info.get("city")), ("CAP", info.get("postcode"))]
            )  # fmt: skip
        except geocode.GeocodingError as exc:
            info_html = f'<div class="finding warn">{ui._e(exc)}</div>'
    formats = ui.kv_table(all_formats(point.lat, point.lon).items(), mono=True)
    card = f'<div class="card"><h3>📍 {point.lat:.6f}, {point.lon:.6f}</h3>{formats}{info_html}</div>'
    links = f'<div class="card"><h3>🔗 Apri in…</h3>{ui.links_html(map_links(point.lat, point.lon))}</div>'
    return card, render_map([Marker(point.lat, point.lon, str(point), "final")], zoom=16, height=380), links


def place_search(query: str):
    if not (query or "").strip():
        return [], ui.empty("Scrivi un luogo, un indirizzo o un nome di attività.", "🔎")
    try:
        results = geocode.search(query.strip(), limit=8)
    except geocode.GeocodingError as exc:
        return [], ui.error_card(str(exc))
    rows = [[r["name"], round(r["lat"], 6), round(r["lon"], 6), r["type"]] for r in results]
    if not rows:
        return [], ui.empty("Nessun risultato.", "🔎")
    return rows, render_map([Marker(r["lat"], r["lon"], r["name"], "other") for r in results], height=380)


def distance_tool(a: str, b: str):
    try:
        p1, p2 = parse_coordinates(a), parse_coordinates(b)
    except CoordinateError as exc:
        return ui.error_card(str(exc))
    d = haversine_km(p1.lat, p1.lon, p2.lat, p2.lon)
    brg = initial_bearing(p1.lat, p1.lon, p2.lat, p2.lon)
    rows = [("Distanza (ortodromica)", f"{d:,.3f} km".replace(",", " ")), ("Azimut iniziale A→B", f"{brg:.1f}° ({compass_point(brg)})")]
    return f'<div class="card"><h3>📏 Distanza e direzione</h3>{ui.kv_table(rows)}</div>'


# ---------------------------------------------------------------------------
# Sun / chronolocation tab
# ---------------------------------------------------------------------------


def _parse_day(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).date()
    return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()


def sun_tool(coords: str, day, time_str: str, tz_name: str):
    try:
        point = parse_coordinates(coords)
        tz = parse_tz(tz_name)
        d = _parse_day(day)
        hh, mm = (int(x) for x in (time_str or "12:00").strip().split(":")[:2])
    except (CoordinateError, ValueError) as exc:
        return ui.error_card(str(exc)), ""
    when = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)
    pos = sun_position(when, point.lat, point.lon)
    times = sun_times(d, point.lat, point.lon, tz)
    ratio = pos.shadow_ratio
    rows = [
        ("Azimut sole", f"{pos.azimuth:.1f}° ({compass_point(pos.azimuth)})"),
        ("Elevazione sole", f"{pos.elevation:.2f}°"),
        ("Direzione ombre", f"{pos.shadow_azimuth:.1f}° ({compass_point(pos.shadow_azimuth)})"),
        ("Rapporto ombra/altezza", f"{ratio:.2f} (un palo di 1 m proietta {ratio:.2f} m)" if ratio else "sole sotto l'orizzonte"),
        ("Alba", ui.fmt_time(times["sunrise"])),
        ("Mezzogiorno solare", ui.fmt_time(times["solar_noon"])),
        ("Tramonto", ui.fmt_time(times["sunset"])),
        ("Ora d'oro", f"fino alle {ui.fmt_time(times['golden_morning_end'])} · dalle {ui.fmt_time(times['golden_evening_start'])}"),
        ("Elevazione massima", f"{times['max_elevation']:.1f}°"),
    ]
    svg = ui.sun_path_svg(day_samples(d, point.lat, point.lon, tz, step_min=10), pos)
    card = f'<div class="card"><h3>☀️ {when:%d/%m/%Y %H:%M} · {ui._e(tz_name or "UTC")}</h3>{ui.kv_table(rows)}</div>'
    return (
        card,
        f'<div class="card" style="text-align:center"><h3>Percorso del sole</h3>{svg}<p class="muted">Pallino giallo: sole · tratteggio: direzione dell\'ombra</p></div>',
    )


def shadow_tool(coords: str, day, tz_name: str, height: float, length: float, shadow_az: float | None):
    try:
        point = parse_coordinates(coords)
        tz = parse_tz(tz_name)
        d = _parse_day(day)
        az = float(shadow_az) if shadow_az not in (None, "", 0) else None
        matches = times_for_shadow(d, point.lat, point.lon, tz, float(height or 0), float(length or 0), az)
    except (CoordinateError, ValueError, TypeError) as exc:
        return ui.error_card(str(exc))
    if not matches:
        return ui.empty("Nessun orario compatibile in quella data: verifica le misure o prova date vicine.", "🌗")
    rows = "".join(
        f"<tr><td><b>{m.when:%H:%M}</b></td><td>{m.elevation:.1f}°</td><td>{m.azimuth:.0f}° {compass_point(m.azimuth)}</td>"
        f"<td>{m.shadow_azimuth:.0f}° {compass_point(m.shadow_azimuth)}</td><td>±{m.error_deg:.2f}°</td></tr>"
        for m in matches
    )
    return (
        '<div class="card"><h3>🕰️ Orari compatibili con l\'ombra</h3><table class="kv wide"><tr><th>Ora</th><th>Elev.</th><th>Sole</th>'
        f"<th>Ombra</th><th>Errore</th></tr>{rows}</table>"
        '<p class="muted">Senza direzione dell\'ombra ci sono di solito due soluzioni (mattino e pomeriggio): misura su mappa '
        "la direzione dell'ombra per scegliere quella giusta.</p></div>"
    )


# ---------------------------------------------------------------------------
# Privacy / compare / batch tabs
# ---------------------------------------------------------------------------


def privacy_tool(image_path: str | None):
    from ..forensics import extract_metadata, strip_metadata

    if not image_path:
        return ui.empty("Carica un'immagine per vedere cosa rivela e ottenerne una copia pulita.", "🧹"), None
    meta = extract_metadata(image_path)
    src = Path(image_path)
    out_dir = Path(tempfile.mkdtemp(prefix="geoosint_clean_"))
    clean = strip_metadata(
        src, out_dir / f"{src.stem}_pulita{src.suffix if src.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} else '.png'}"
    )
    after = extract_metadata(clean)
    summary = (
        '<div class="card"><h3>🧹 Prima → dopo</h3>'
        + ui.kv_table([
            ("Tag EXIF", f"{len(meta.exif)} → {len(after.exif)}"),
            ("XMP", f"{len(meta.xmp)} → {len(after.xmp)}"),
            ("GPS", f"{'presente' if meta.gps else 'assente'} → {'presente' if after.gps else 'assente'}"),
            ("Fotocamera", f"{meta.camera or '—'} → {after.camera or '—'}"),
        ])
        + "</div>"
    )  # fmt: skip
    return ui.metadata_card(meta) + summary, str(clean)


def compare_tool(a: str | None, b: str | None):
    from ..forensics import compute_hashes, extract_metadata, hamming

    if not a or not b:
        return ui.empty("Carica due immagini da confrontare.", "🧩")
    ha, hb = compute_hashes(a), compute_hashes(b)
    if ha.sha256 == hb.sha256:
        verdict = ("good", "File identici bit a bit (stesso SHA-256).")
    else:
        dist = min(hamming(ha.phash, hb.phash), hamming(ha.dhash, hb.dhash))
        if dist <= 6:
            verdict = (
                "good",
                f"Quasi certamente la stessa immagine (distanza percettiva {dist}/64): ricompressa, ridimensionata o ritagliata leggermente.",
            )
        elif dist <= 14:
            verdict = ("warn", f"Immagini molto simili (distanza {dist}/64): possibile versione modificata o scatto ravvicinato.")
        else:
            verdict = ("bad", f"Immagini diverse (distanza {dist}/64).")
    ma, mb = extract_metadata(a), extract_metadata(b)
    rows = "".join(
        f"<tr><th>{ui._e(label)}</th><td>{ui._e(x)}</td><td>{ui._e(y)}</td></tr>"
        for label, x, y in [
            ("Dimensioni", f"{ma.width}×{ma.height}", f"{mb.width}×{mb.height}"),
            ("Formato", ma.format, mb.format),
            ("Peso", f"{ma.file_size / 1024:.0f} KB", f"{mb.file_size / 1024:.0f} KB"),
            ("Fotocamera", ma.camera or "—", mb.camera or "—"),
            ("Data scatto", ma.datetime_original or "—", mb.datetime_original or "—"),
            ("Software", ma.software or "—", mb.software or "—"),
            ("Qualità JPEG", ma.jpeg_quality or "—", mb.jpeg_quality or "—"),
            ("GPS", f"{ma.gps.lat:.5f}, {ma.gps.lon:.5f}" if ma.gps else "—", f"{mb.gps.lat:.5f}, {mb.gps.lon:.5f}" if mb.gps else "—"),
            ("pHash", ha.phash, hb.phash),
            ("SHA-256", ha.sha256[:16] + "…", hb.sha256[:16] + "…"),
        ]
    )
    hint = ""
    if ma.file_size and mb.file_size and verdict[0] != "bad":
        older = "A" if (ma.exif and not mb.exif) or ma.width * ma.height > mb.width * mb.height else "B"
        hint = (
            f'<p class="muted">Indizio di provenienza: la versione {older} ha più metadati/risoluzione ed è più vicina all\'originale.</p>'
        )
    return (
        f'<div class="card"><h3>🧩 Confronto</h3><div class="finding {verdict[0]}">{ui._e(verdict[1])}</div>'
        f'<table class="kv wide" style="margin-top:10px"><tr><th></th><th>A</th><th>B</th></tr>{rows}</table>{hint}</div>'
    )


def batch_tool(files: list | None, modules: list[str], preset: str, progress=None):
    if not files:
        return [], empty_map(), None
    enabled = {MODULES[m] for m in modules if m in MODULES}
    opts = AnalysisOptions(
        metadata="metadata" in enabled,
        forensics="forensics" in enabled,
        geoclip="geoclip" in enabled,
        ocr="ocr" in enabled,
        yolo="yolo" in enabled,
        ai="ai" in enabled,
        reverse_geocode="reverse_geocode" in enabled,
        geoclip_preset=preset,
    )
    pipe = Pipeline()
    rows, markers, dump = [], [], []
    paths = [getattr(f, "name", f) for f in files]
    for i, path in enumerate(paths):
        name = Path(path).name
        if progress is not None:
            progress(i / len(paths), desc=f"{i + 1}/{len(paths)} · {name}")
        try:
            r = pipe.run(path, opts)
        except Exception as exc:
            rows.append([name, "", "", "errore", "", "", "", str(exc)[:120]])
            continue
        dump.append(r.to_dict())
        f = r.final
        m = r.metadata
        rows.append(
            [
                name,
                round(f.point.lat, 6) if f else "",
                round(f.point.lon, 6) if f else "",
                f.source.label if f else "—",
                f.confidence if f else "",
                (m.camera if m else "") or "",
                (m.datetime_original if m else "") or "",
                r.forensics.hashes.sha256[:16] if r.forensics and r.forensics.hashes else "",
            ]
        )
        if f:
            markers.append(
                Marker(f.point.lat, f.point.lon, f"{name} · {f.source.label}", "geoclip" if f.source.value == "geoclip" else f.source.value)
            )
    out_dir = get_settings().reports_dir / f"batch_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "riepilogo.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(BATCH_HEADERS)
        writer.writerows(rows)
    json_path = out_dir / "risultati.json"
    json_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, render_map(markers) if markers else empty_map(), [str(csv_path), str(json_path)]


BATCH_HEADERS = ["File", "Lat", "Lon", "Fonte", "Affidabilità", "Fotocamera", "Data scatto", "SHA-256"]


def status_html() -> str:
    s = get_settings()
    engine_rows = "".join(
        f"<tr><th>{ui._e(st.name)}</th><td>{'🟢 caricato' if st.loaded else ('⚪ pronto (caricamento al primo uso)' if st.installed else '🔴 ' + ui._e(st.detail))}</td></tr>"
        for st in statuses()
    )
    gemini_ok, gemini_detail = modules_available("google.genai")
    engine_rows += f"<tr><th>Gemini</th><td>{'🟢 chiave configurata' if s.gemini_api_key else '🟡 chiave assente (inseriscila nelle opzioni o in .env)'}{'' if gemini_ok else ' · 🔴 ' + ui._e(gemini_detail)}</td></tr>"
    settings_rows = ui.kv_table(
        [
            ("Modello Gemini", s.gemini_model),
            ("Ricerca Google (grounding)", "attiva" if s.gemini_search else "disattiva"),
            ("Preset GeoCLIP", s.geoclip_preset),
            ("Lingue OCR", ", ".join(s.ocr_languages)),
            ("Modello YOLO", s.yolo_model),
            ("Ricerche online", "attive" if s.online_lookups else "disattivate"),
            ("Cartella report", str(s.reports_dir)),
            ("Limite immagine", f"{s.max_image_mb} MB"),
        ]
    )
    return (
        f'<div class="card"><h3>⚙️ Motori</h3><table class="kv">{engine_rows}</table></div>'
        f'<div class="card"><h3>🔧 Configurazione attiva</h3>{settings_rows}'
        '<p class="muted">Modifica tramite variabili d\'ambiente o file <code>.env</code> (vedi <code>.env.example</code>).</p></div>'
    )
