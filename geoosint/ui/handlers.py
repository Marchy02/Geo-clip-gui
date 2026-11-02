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
        # ok = loaded, idle = installed (loads on first use), bad = missing
        chips.append((st.name, "ok" if st.loaded else ("idle" if st.installed else "bad")))
    chips.append(("Gemini", "ok" if s.gemini_api_key and modules_available("google.genai")[0] else "warn"))
    chips.append(("Rete" if s.online_lookups else "Offline", "ok" if s.online_lookups else "warn"))
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
        return blank_outputs()

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
        return blank_outputs(str(exc), error=True)

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
        geo_html = ui.formats_card(final.point.lat, final.point.lon) + ui.section(
            "Verifica sul posto", ui.links_html(map_links(final.point.lat, final.point.lon)), "Apri il punto del verdetto in"
        )
    else:
        geo_html = ui.empty("Senza un punto non ci sono coordinate da convertire.", "Nessuna coordinata")

    geo_rows = [
        [i + 1, round(g.lat, 5), round(g.lon, 5), f"{(g.prob or 0) * 100:.2f}%", g.address or ""] for i, g in enumerate(report.geoclip)
    ]
    exif_rows = [[k, v] for k, v in (report.metadata.exif.items() if report.metadata else [])]
    exif_rows += [[f"XMP · {k}", v] for k, v in (report.metadata.xmp.items() if report.metadata else [])]
    exif_rows += [[f"PNG · {k}", v] for k, v in (report.metadata.png_text.items() if report.metadata else [])]

    from ..engines.yolo_engine import GEO_RELEVANT, summarize

    reverse = ui.links_html(reverse_image_links(image_url.strip() or None)) + ui.links_html(VERIFICATION_TOOLS)
    intro = (
        "I link cercano direttamente l'immagine dall'URL che hai indicato."
        if image_url.strip()
        else "L'immagine è sul tuo computer: apri un motore e carica il file. Se indichi un URL pubblico, i link cercano direttamente."
    )
    reverse_html = ui.section("Ricerca inversa e verifica", f'<p class="muted">{ui._e(intro)}</p>{reverse}', "Trova la copia più vecchia")

    fz = report.forensics
    return (
        ui.verdict_card(report),
        map_with_legend(_markers(report)),
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


def map_with_legend(markers: list[Marker], height: int = 460) -> str:
    kinds = {m.kind for m in markers}
    return ui.map_legend(kinds) + (render_map(markers, height=height) if markers else empty_map(height))


def blank_outputs(message: str = "", error: bool = False):
    if error:
        card = ui.error_card(message, "Analisi non riuscita")
    elif message:
        card = ui.empty(message)
    else:
        card = ui.verdict_placeholder()
    return (card, map_with_legend([]), [], "", "", "", [], None, None, "", "", None, "", "", None, "", None)


def fetch_url(url: str):
    from ..fetch import FetchError, download_image

    if not (url or "").strip():
        return None, ui.error_card("Incolla l'indirizzo completo di un'immagine, che inizi con https://.", "URL mancante")
    try:
        path = download_image(url)
    except FetchError as exc:
        return None, ui.error_card(str(exc), "Download non riuscito")
    return str(path), ui.notice(f"Immagine scaricata: {path.name}. Ora premi Analizza immagine.", "ok")


# ---------------------------------------------------------------------------
# Coordinates tab
# ---------------------------------------------------------------------------


def coordinate_tool(text: str, do_reverse: bool):
    try:
        point = parse_coordinates(text)
    except CoordinateError as exc:
        return ui.error_card(str(exc), "Coordinate non riconosciute"), map_with_legend([], 380), ""
    info_html = ""
    if do_reverse and get_settings().online_lookups:
        try:
            info = geocode.reverse_geocode(point.lat, point.lon, zoom=18)
            info_html = ui.kv_table(
                [("Indirizzo", info.get("display_name")), ("Paese", f"{info.get('country', '')} {info.get('country_code', '')}".strip()),
                 ("Regione", info.get("state")), ("Città", info.get("city")), ("CAP", info.get("postcode"))]
            )  # fmt: skip
        except geocode.GeocodingError as exc:
            info_html = ui.notice(str(exc), "warn")
    formats = ui.kv_table(all_formats(point.lat, point.lon).items(), mono=True)
    card = ui.section(f"{point.lat:.6f}, {point.lon:.6f}", formats + info_html, "Punto")
    links = ui.section("Apri il punto in", ui.links_html(map_links(point.lat, point.lon)))
    marker = Marker(point.lat, point.lon, str(point), "final")
    return card, ui.map_legend({"final"}) + render_map([marker], zoom=16, height=380), links


def place_search(query: str):
    if not (query or "").strip():
        return [], ui.empty("Scrivi un indirizzo, un monumento o il nome di un'attività.", "Cerca un luogo")
    try:
        results = geocode.search(query.strip(), limit=8)
    except geocode.GeocodingError as exc:
        return [], ui.error_card(str(exc), "Ricerca non disponibile")
    rows = [[r["name"], round(r["lat"], 6), round(r["lon"], 6), r["type"]] for r in results]
    if not rows:
        return [], ui.empty("OpenStreetMap non ha trovato nulla: prova con meno parole o aggiungi la città.", "Nessun risultato")
    return rows, render_map([Marker(r["lat"], r["lon"], r["name"], "other") for r in results], height=380)


def distance_tool(a: str, b: str):
    try:
        p1, p2 = parse_coordinates(a), parse_coordinates(b)
    except CoordinateError as exc:
        return ui.error_card(str(exc), "Coordinate non riconosciute")
    d = haversine_km(p1.lat, p1.lon, p2.lat, p2.lon)
    brg = initial_bearing(p1.lat, p1.lon, p2.lat, p2.lon)
    rows = [("Distanza (ortodromica)", f"{d:,.3f} km".replace(",", " ")), ("Azimut iniziale A→B", f"{brg:.1f}° ({compass_point(brg)})")]
    return ui.section("Distanza e direzione", ui.kv_table(rows))


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
        return ui.error_card(str(exc), "Dati non validi"), ""
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
    card = ui.section(f"{when:%d/%m/%Y %H:%M}", ui.kv_table(rows), f"Sole · {tz_name or 'UTC'}")
    figure = (
        f'<figure class="sun-figure">{svg}<figcaption>Percorso del sole dall\'alba al tramonto, con le ore. '
        "Cerchio: posizione all'ora scelta. Tratteggio: direzione delle ombre.</figcaption></figure>"
    )
    return card, figure


def shadow_tool(coords: str, day, tz_name: str, height: float, length: float, shadow_az: float | None):
    try:
        point = parse_coordinates(coords)
        tz = parse_tz(tz_name)
        d = _parse_day(day)
        az = float(shadow_az) if shadow_az not in (None, "", 0) else None
        matches = times_for_shadow(d, point.lat, point.lon, tz, float(height or 0), float(length or 0), az)
    except (CoordinateError, ValueError, TypeError) as exc:
        return ui.error_card(str(exc), "Dati non validi")
    if not matches:
        return ui.empty(
            "In quella data il sole non raggiunge l'altezza che queste misure richiedono. Controlla altezza e ombra, o prova date vicine.",
            "Nessun orario compatibile",
        )
    rows = [
        [f"{m.when:%H:%M}", f"{m.elevation:.1f}°", f"{m.azimuth:.0f}° {compass_point(m.azimuth)}",
         f"{m.shadow_azimuth:.0f}° {compass_point(m.shadow_azimuth)}", f"±{m.error_deg:.2f}°"]
        for m in matches
    ]  # fmt: skip
    note = (
        "<p class='muted'>Senza la direzione dell'ombra di solito ci sono due soluzioni, mattino e pomeriggio. "
        "Misura la direzione sul satellite per scegliere quella giusta.</p>"
    )
    table = ui.grid_table(["Ora", "Elevazione", "Sole", "Ombra verso", "Scarto"], rows, mono_cols=(0,))
    return ui.section("Orari compatibili con l'ombra", table + note)


# ---------------------------------------------------------------------------
# Privacy / compare / batch tabs
# ---------------------------------------------------------------------------


def privacy_tool(image_path: str | None):
    from ..forensics import extract_metadata, strip_metadata

    if not image_path:
        return ui.empty("Carica una foto: vedrai cosa rivela e potrai scaricarne una copia senza metadati.", "Nessuna immagine"), None
    meta = extract_metadata(image_path)
    src = Path(image_path)
    out_dir = Path(tempfile.mkdtemp(prefix="geoosint_clean_"))
    clean = strip_metadata(
        src, out_dir / f"{src.stem}_pulita{src.suffix if src.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} else '.png'}"
    )
    after = extract_metadata(clean)
    summary = ui.section("Copia pulita pronta", ui.kv_table([
        ("Tag EXIF", f"{len(meta.exif)} → {len(after.exif)}"),
        ("XMP", f"{len(meta.xmp)} → {len(after.xmp)}"),
        ("GPS", f"{'presente' if meta.gps else 'assente'} → {'presente' if after.gps else 'assente'}"),
        ("Fotocamera", f"{meta.camera or '—'} → {after.camera or '—'}"),
    ]), "Prima → dopo")  # fmt: skip
    return summary + ui.metadata_card(meta), str(clean)


def compare_tool(a: str | None, b: str | None):
    from ..forensics import compute_hashes, extract_metadata, hamming

    if not a or not b:
        return ui.empty("Carica entrambe le immagini, A e B, poi premi Confronta immagini.", "Servono due immagini")
    ha, hb = compute_hashes(a), compute_hashes(b)
    if ha.sha256 == hb.sha256:
        verdict = ("ok", "File identici bit a bit (stesso SHA-256).")
    else:
        dist = min(hamming(ha.phash, hb.phash), hamming(ha.dhash, hb.dhash))
        if dist <= 6:
            verdict = (
                "ok",
                f"Quasi certamente la stessa immagine (distanza percettiva {dist}/64): ricompressa, ridimensionata o ritagliata leggermente.",
            )
        elif dist <= 14:
            verdict = ("warn", f"Immagini molto simili (distanza {dist}/64): possibile versione modificata o scatto ravvicinato.")
        else:
            verdict = ("bad", f"Immagini diverse (distanza {dist}/64).")
    ma, mb = extract_metadata(a), extract_metadata(b)
    rows = [
        [label, x, y]
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
    ]
    hint = ""
    if ma.file_size and mb.file_size and verdict[0] != "bad":
        older = "A" if (ma.exif and not mb.exif) or ma.width * ma.height > mb.width * mb.height else "B"
        hint = (
            f'<p class="muted">Indizio di provenienza: la versione {older} ha più metadati/risoluzione ed è più vicina all\'originale.</p>'
        )
    return ui.section("Esito del confronto", ui.notice(verdict[1], verdict[0])) + ui.section(
        "Metadati a confronto", ui.grid_table(["", "A", "B"], rows) + hint
    )


def batch_tool(files: list | None, modules: list[str], preset: str, progress=None):
    if not files:
        return [], map_with_legend([], 400), None
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
    return rows, map_with_legend(markers, 400), [str(csv_path), str(json_path)]


BATCH_HEADERS = ["File", "Lat", "Lon", "Fonte", "Affidabilità", "Fotocamera", "Data scatto", "SHA-256"]


def status_html() -> str:
    s = get_settings()
    rows = []
    for st in statuses():
        state = "pronto, si carica al primo uso" if st.installed else st.detail
        rows.append((st.name, "caricato" if st.loaded else state))
    gemini_ok, gemini_detail = modules_available("google.genai")
    if not gemini_ok:
        rows.append(("Gemini", gemini_detail))
    else:
        rows.append(("Gemini", "chiave configurata" if s.gemini_api_key else "chiave assente: aggiungila nelle opzioni o nel file .env"))
    settings_rows = ui.kv_table(
        [
            ("Modello Gemini", s.gemini_model),
            ("Verifica con Google Search", "attiva" if s.gemini_search else "disattiva"),
            ("Modello GeoCLIP", s.geoclip_preset),
            ("Lingue OCR", ", ".join(s.ocr_languages)),
            ("Modello YOLO", s.yolo_model),
            ("Ricerche online", "attive" if s.online_lookups else "disattivate"),
            ("Cartella dei report", str(s.reports_dir)),
            ("Dimensione massima immagine", f"{s.max_image_mb} MB"),
        ]
    )
    return ui.section("Motori", ui.kv_table(rows), "Stato") + ui.section(
        "Configurazione attiva",
        settings_rows
        + "<p class='muted'>Si cambia con le variabili d'ambiente o il file <code>.env</code> (modello in <code>.env.example</code>).</p>",
    )
