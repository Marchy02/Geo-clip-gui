"""HTML fragments for the dashboard. Every dynamic value goes through ``_e``."""

from __future__ import annotations

import html
import math
from collections.abc import Iterable
from datetime import datetime

from ..geo.coords import all_formats
from ..geo.links import Link
from ..geo.sun import SunPosition
from ..models import AiVerdict, AnalysisReport, FinalLocation, ForensicsResult, MetadataReport, OcrText, StepStatus


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def empty(message: str, icon: str = "🛰️") -> str:
    return f'<div class="empty"><span class="ico">{icon}</span>{_e(message)}</div>'


def error_card(message: str) -> str:
    return f'<div class="card"><div class="finding bad">❌ {_e(message)}</div></div>'


def kv_table(rows: Iterable[tuple[str, object]], mono: bool = False) -> str:
    body = "".join(
        f"<tr><th>{_e(k)}</th><td>{f'<code>{_e(v)}</code>' if mono else _e(v)}</td></tr>" for k, v in rows if v not in (None, "")
    )
    return f'<table class="kv">{body}</table>' if body else ""


def links_html(links: Iterable[Link], grouped: bool = True) -> str:
    links = list(links)
    if not links:
        return ""

    def anchors(items):
        return "".join(
            f'<a href="{_e(link.url)}" target="_blank" rel="noopener noreferrer" title="{_e(link.note)}">{_e(link.name)}</a>'
            for link in items
        )

    if not grouped:
        return f'<div class="links">{anchors(links)}</div>'
    groups: dict[str, list[Link]] = {}
    for link in links:
        groups.setdefault(link.category, []).append(link)
    return "".join(
        f'<div class="linkgroup"><h4>{_e(cat)}</h4><div class="links">{anchors(items)}</div></div>' for cat, items in groups.items()
    )


def chip(text: str, state: str = "") -> str:
    return f'<span class="chip {state}"><span class="dot"></span>{_e(text)}</span>'


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def header(version: str, statuses: list[tuple[str, str]]) -> str:
    chips = "".join(chip(name, state) for name, state in statuses)
    return (
        '<div class="go-header"><div class="go-brand"><div class="go-logo">🛰️</div><div>'
        f'<div class="go-title">GeoOSINT <span class="muted" style="font-size:.85rem;font-weight:500">v{_e(version)}</span></div>'
        '<div class="go-sub">Geolocalizzazione · Metadati · Forense · OCR · Oggetti · Cronolocalizzazione · Pivot</div>'
        f'</div></div><div class="go-chips">{chips}</div></div>'
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def verdict_card(report: AnalysisReport) -> str:
    final: FinalLocation | None = report.final
    if not final:
        return (
            '<div class="card verdict"><h3>Nessuna posizione determinata</h3>'
            '<p class="muted">Nessun GPS nei metadati e nessun motore di geolocalizzazione disponibile. '
            "Controlla la scheda <b>Log</b> e attiva GeoCLIP o l'arbitro IA.</p></div>"
        )
    p = final.point
    prob = f"{p.prob * 100:.0f}%" if p.prob is not None else "—"
    ocr_count = len(report.ocr)
    obj_count = len(report.detections)
    kpis = [
        (final.source.label, "fonte"),
        (f'<span class="conf-{final.confidence}">{final.confidence.upper()}</span>', "affidabilità"),
        (prob, "probabilità"),
        (f"{ocr_count} / {obj_count}", "testi / oggetti"),
    ]
    if report.geoclip_spread_km is not None:
        kpis.append((f"~{report.geoclip_spread_km:.0f} km", "dispersione GeoCLIP"))
    kpi_html = "".join(f"<div class='kpi'><b>{v if '<span' in v else _e(v)}</b><span class='lbl'>{_e(k)}</span></div>" for v, k in kpis)
    address = f'<div class="muted">{_e(p.address)}</div>' if p.address else ""
    return (
        f'<div class="card verdict {final.source.value}">'
        f'<div class="muted" style="text-transform:uppercase;letter-spacing:.08em;font-size:.72rem">Verdetto</div>'
        f'<div class="place">{_e(p.label)}</div><div class="coords">{p.lat:.6f}, {p.lon:.6f}</div>{address}'
        f'<div class="why">{_e(final.explanation)}</div><div class="kpis">{kpi_html}</div></div>'
    )


def formats_card(lat: float, lon: float) -> str:
    return f'<div class="card"><h3>📐 Coordinate in tutti i formati</h3>{kv_table(all_formats(lat, lon).items(), mono=True)}</div>'


def ai_card(ai: AiVerdict | None) -> str:
    if ai is None:
        return empty("Arbitro IA non eseguito. Inserisci una chiave Gemini nelle opzioni per attivarlo.", "🤖")
    if ai.error:
        return error_card(ai.error)
    bar = f'<div class="bar"><i style="width:{ai.confidence}%"></i></div>'
    head = (
        f"<h3>🤖 {_e(ai.location_name or 'Località non determinata')}</h3>"
        f'<div class="muted">{_e(ai.country or "paese n/d")} · confidenza {ai.confidence}% · '
        f"{'override con prove indipendenti' if ai.override else 'nessun override'} · {_e(ai.model)}</div>{bar}"
    )
    coords = f'<p class="mono">{ai.lat:.6f}, {ai.lon:.6f}</p>' if ai.lat is not None and ai.lon is not None else ""
    clues = ("<h4>Indizi</h4><ul>" + "".join(f"<li>{_e(c)}</li>" for c in ai.clues) + "</ul>") if ai.clues else ""
    reasoning = f"<h4>Ragionamento</h4><p>{_e(ai.reasoning)}</p>" if ai.reasoning else ""
    alts = ("<h4>Ipotesi alternative</h4><ul>" + "".join(f"<li>{_e(c)}</li>" for c in ai.alternatives) + "</ul>") if ai.alternatives else ""
    sources = (
        "<h4>Fonti web consultate</h4><ul>"
        + "".join(
            f'<li><a href="{_e(s.get("url"))}" target="_blank" rel="noopener noreferrer">{_e(s.get("title"))}</a></li>' for s in ai.sources
        )
        + "</ul>"
        if ai.sources
        else ""
    )
    return f'<div class="card">{head}{coords}{clues}{reasoning}{alts}{sources}</div>'


def _finding(text: str) -> str:
    low = text.lower()
    level = "bad" if text.startswith("⚠") or "discordanza" in low or "generato da ia" in low else "warn"
    if "nessuna anomalia" in low:
        level = "good"
    return f'<div class="finding {level}">{_e(text)}</div>'


def metadata_card(meta: MetadataReport | None) -> str:
    if meta is None:
        return empty("Analisi dei metadati non eseguita.", "🗂️")
    gps = meta.gps
    rows = [
        ("File", meta.file_name),
        ("Formato", f"{meta.format} · {meta.mime}" if meta.mime else meta.format),
        ("Dimensioni", f"{meta.width} × {meta.height} px · {meta.mode}"),
        ("Peso", f"{meta.file_size / 1024:.1f} KB"),
        ("Fotocamera", meta.camera),
        ("Obiettivo", meta.lens),
        ("Software", meta.software),
        ("Data scatto", " ".join(x for x in (meta.datetime_original, meta.timezone_offset) if x)),
        ("Qualità JPEG stimata", meta.jpeg_quality),
        ("Profilo ICC", meta.icc_profile),
        ("Miniatura EXIF", "presente" if meta.has_thumbnail else ""),
    ]
    if gps:
        rows += [
            ("GPS", f"{gps.lat:.6f}, {gps.lon:.6f}"),
            ("Altitudine", f"{gps.altitude_m} m" if gps.altitude_m is not None else ""),
            ("Direzione scatto", f"{gps.direction_deg:.0f}° ({gps.direction_ref or '?'})" if gps.direction_deg is not None else ""),
            ("Ora GPS (UTC)", gps.timestamp_utc),
        ]
    observations = [*meta.ai_markers, *meta.findings]
    obs_html = "".join(_finding(x) for x in observations) or _finding("Nessuna osservazione particolare.")
    return f'<div class="card"><h3>🗂️ Riepilogo</h3>{kv_table(rows)}<h4>Osservazioni</h4>{obs_html}</div>'


def forensics_card(fz: ForensicsResult | None) -> str:
    if fz is None or fz.hashes is None:
        return empty("Analisi forense non eseguita.", "🧪")
    h = fz.hashes
    rows = [
        ("SHA-256", h.sha256),
        ("SHA-1", h.sha1),
        ("MD5", h.md5),
        ("pHash", h.phash),
        ("dHash", h.dhash),
        ("aHash", h.ahash),
        ("ELA medio", f"{fz.ela_score:.2f}" if fz.ela_score is not None else ""),
    ]
    findings = "".join(_finding(x) for x in fz.findings)
    tips = (
        '<p class="muted" style="margin-top:10px">ELA: le aree ritoccate tendono a brillare più del resto con bordi netti. '
        "Rumore: zone con grana diversa dal contesto possono essere state incollate o generate. Sono indizi, non prove.</p>"
    )
    return f'<div class="card"><h3>🧪 Impronte e indicatori</h3>{kv_table(rows, mono=True)}{findings}{tips}</div>'


def ocr_card(texts: list[OcrText]) -> str:
    if not texts:
        return empty("Nessun testo leggibile rilevato (o OCR non eseguito).", "🔤")
    from ..geo.links import text_search_links

    items = []
    for t in texts:
        pivots = links_html(text_search_links(t.text)[:4], grouped=False)
        items.append(
            f'<div class="ocr-item"><div><code>{_e(t.text)}</code> <span class="muted">{t.confidence:.0%}</span></div>{pivots}</div>'
        )
    return f'<div class="card"><h3>🔤 Testo rilevato</h3>{"".join(items)}</div>'


def objects_card(counts: dict[str, int], relevant: set[str]) -> str:
    if not counts:
        return empty("Nessun oggetto rilevato (o YOLO non eseguito).", "🚦")
    chips = "".join(chip(f"{label} × {n}", "ok" if label in relevant else "") for label, n in counts.items())
    return f'<div class="card"><h3>🚦 Oggetti</h3><div class="steps">{chips}</div><p class="muted" style="margin-top:8px">In verde le classi utili alla geolocalizzazione (veicoli, segnaletica, infrastrutture).</p></div>'


def steps_html(steps: list[StepStatus]) -> str:
    if not steps:
        return ""
    state = {"ok": "ok", "skipped": "warn", "error": "bad"}
    chips = "".join(chip(f"{s.name} · {s.duration_ms} ms", state.get(s.status, "")) for s in steps)
    details = "".join(f"<li><b>{_e(s.name)}</b>: {_e(s.detail)}</li>" for s in steps if s.detail)
    return (
        f'<div class="card"><h3>📋 Esecuzione</h3><div class="steps">{chips}</div>'
        + (f"<ul style='margin-top:10px'>{details}</ul>" if details else "")
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Sun tools
# ---------------------------------------------------------------------------


def sun_path_svg(samples: list[SunPosition], marker: SunPosition | None = None, size: int = 300) -> str:
    """Polar sun-path diagram: azimuth around, elevation from horizon (edge) to zenith (centre)."""
    c = size / 2
    r = size / 2 - 22

    def xy(az: float, el: float) -> tuple[float, float]:
        rr = r * (90 - max(el, 0)) / 90
        a = math.radians(az)
        return c + rr * math.sin(a), c - rr * math.cos(a)

    rings = "".join(
        f'<circle cx="{c}" cy="{c}" r="{r * (90 - e) / 90:.1f}" fill="none" stroke="currentColor" stroke-opacity=".15"/>'
        for e in (0, 30, 60)
    )
    labels = "".join(
        f'<text x="{c + (r + 12) * math.sin(math.radians(a)):.1f}" y="{c - (r + 12) * math.cos(math.radians(a)) + 4:.1f}" '
        f'text-anchor="middle" font-size="11" fill="currentColor" fill-opacity=".7">{t}</text>'
        for a, t in ((0, "N"), (90, "E"), (180, "S"), (270, "O"))
    )
    above = [s for s in samples if s.elevation > 0]
    path = ""
    if above:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(s.azimuth, s.elevation) for s in above))
        path = f'<polyline points="{pts}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round"/>'
        hours = ""
        for s in above:
            if s.when.minute == 0:
                x, y = xy(s.azimuth, s.elevation)
                hours += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#f59e0b"/><text x="{x + 5:.1f}" y="{y - 4:.1f}" font-size="9" fill="currentColor" fill-opacity=".7">{s.when.hour}</text>'
        path += hours
    mark = ""
    if marker and marker.elevation > 0:
        x, y = xy(marker.azimuth, marker.elevation)
        sx, sy = xy(marker.shadow_azimuth, 60)
        mark = (
            f'<line x1="{c}" y1="{c}" x2="{sx:.1f}" y2="{sy:.1f}" stroke="#64748b" stroke-width="3" stroke-dasharray="4 3"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#facc15" stroke="#f59e0b" stroke-width="2"/>'
        )
    return (
        f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:{size}px;color:var(--body-text-color)" role="img" '
        f'aria-label="Percorso del sole">{rings}<line x1="{c}" y1="{c - r}" x2="{c}" y2="{c + r}" stroke="currentColor" stroke-opacity=".12"/>'
        f'<line x1="{c - r}" y1="{c}" x2="{c + r}" y2="{c}" stroke="currentColor" stroke-opacity=".12"/>{labels}{path}{mark}</svg>'
    )


def fmt_time(value: datetime | None) -> str:
    return value.strftime("%H:%M") if isinstance(value, datetime) else "—"
