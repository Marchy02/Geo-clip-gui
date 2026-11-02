"""HTML fragments for the dashboard. Every dynamic value goes through ``_e``.

Styling lives in ``theme.py``; these helpers only emit semantic markup with
the classes defined there (``sec``, ``kv``, ``pill``, ``finding``...).
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable
from datetime import datetime

from ..geo.coords import all_formats, to_dms
from ..geo.links import Link
from ..geo.sun import SunPosition
from ..models import AiVerdict, AnalysisReport, ForensicsResult, MetadataReport, OcrText, StepStatus

STEP_STATE = {"ok": "ok", "skipped": "warn", "error": "bad"}
STEP_LABEL = {"ok": "eseguito", "skipped": "saltato", "error": "errore"}

MARK_SVG = (
    '<svg viewBox="0 0 40 40" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.4">'
    '<circle cx="20" cy="20" r="13"/><circle cx="20" cy="20" r="3" fill="currentColor" stroke="none"/>'
    '<path d="M20 2v9M20 29v9M2 20h9M29 20h9"/></svg>'
)


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


# ---------------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------------


def empty(message: str, title: str = "") -> str:
    head = f"<b>{_e(title)}</b>" if title else ""
    return f'<div class="state">{head}<span>{_e(message)}</span></div>'


def error_card(message: str, title: str = "Operazione non riuscita") -> str:
    return f'<div class="state error" role="alert"><b>{_e(title)}</b><span>{_e(message)}</span></div>'


def section(title: str, body: str, eyebrow: str = "") -> str:
    eb = f'<div class="eyebrow">{_e(eyebrow)}</div>' if eyebrow else ""
    return f'<section class="sec">{eb}<h3>{_e(title)}</h3>{body}</section>'


def notice(text: str, level: str = "ok") -> str:
    return f'<div class="finding {level}">{_e(text)}</div>'


def kv_table(rows: Iterable[tuple[str, object]], mono: bool = False) -> str:
    body = "".join(
        f"<tr><th scope='row'>{_e(k)}</th><td>{f'<code>{_e(v)}</code>' if mono else _e(v)}</td></tr>"
        for k, v in rows
        if v not in (None, "")
    )
    return f'<div class="kv-wrap"><table class="kv">{body}</table></div>' if body else ""


def grid_table(headers: list[str], rows: list[list[object]], mono_cols: tuple[int, ...] = ()) -> str:
    head = "".join(f"<th scope='col'>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td class='{'mono' if i in mono_cols else ''}'>{_e(c)}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    return f'<div class="kv-wrap"><table class="kv grid"><tr>{head}</tr>{body}</table></div>'


def pill(text: str, state: str = "") -> str:
    return f'<span class="pill {state}"><i></i>{_e(text)}</span>'


# Kept for backwards compatibility with earlier callers.
chip = pill


def links_html(links: Iterable[Link], grouped: bool = True) -> str:
    links = list(links)
    if not links:
        return ""
    if not grouped:
        anchors = "".join(f'<a href="{_e(link.url)}" target="_blank" rel="noopener noreferrer">{_e(link.name)}</a>' for link in links)
        return f'<span class="inline-links">{anchors}</span>'
    groups: dict[str, list[Link]] = {}
    for link in links:
        groups.setdefault(link.category, []).append(link)
    cols = []
    for cat, items in groups.items():
        lis = "".join(
            f'<li><a href="{_e(link.url)}" target="_blank" rel="noopener noreferrer"><span class="nm">{_e(link.name)}</span>'
            + (f"<small>{_e(link.note)}</small>" if link.note else "")
            + "</a></li>"
            for link in items
        )
        cols.append(f'<div class="linkgroup"><div class="eyebrow">{_e(cat)}</div><ul>{lis}</ul></div>')
    return f'<div class="linkgroups">{"".join(cols)}</div>'


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------


def header(version: str, statuses: list[tuple[str, str]]) -> str:
    pills = "".join(pill(name, state) for name, state in statuses)
    return (
        '<header class="go-mast"><div class="go-mark">'
        f"{MARK_SVG}<div><div class='go-word'>GeoOSINT<small>v{_e(version)}</small></div>"
        "<div class='go-tag'>Dove è stata scattata questa foto, e come dimostrarlo.</div></div></div>"
        f'<div class="go-engines"><span class="lbl">Motori</span>{pills}</div></header>'
    )


# ---------------------------------------------------------------------------
# Analysis results
# ---------------------------------------------------------------------------


def run_line(steps: list[StepStatus]) -> str:
    if not steps:
        return ""
    pills = "".join(pill(f"{s.name} · {STEP_LABEL.get(s.status, s.status)}", STEP_STATE.get(s.status, "")) for s in steps)
    return f'<div class="runline"><span class="lbl">Esecuzione</span>{pills}</div>'


def verdict_placeholder() -> str:
    return (
        '<div class="verdict"><div><div class="eyebrow">Verdetto</div><div class="place">In attesa di un\'immagine</div></div>'
        '<p class="why">Carica una foto a sinistra, oppure prova l\'esempio incluso, e premi <b>Analizza immagine</b>. '
        "Qui comparirà il punto più affidabile con la fonte, il livello di affidabilità e il motivo della scelta: "
        "GPS nei metadati, poi arbitro IA con prove indipendenti, poi stima GeoCLIP.</p></div>"
    )


def verdict_card(report: AnalysisReport) -> str:
    final = report.final
    if not final:
        return (
            '<div class="verdict"><div><div class="eyebrow">Verdetto</div>'
            '<div class="place">Posizione non determinata</div></div>'
            '<p class="why">Nessun GPS nei metadati e nessun motore di geolocalizzazione ha prodotto un risultato. '
            "La scheda Registro indica quali moduli sono stati saltati e perché.</p>"
            f"{run_line(report.steps)}</div>"
        )
    p = final.point
    facts = [
        ("Fonte", _e(final.source.label)),
        (
            "Affidabilità",
            f'<span class="lvl-{final.confidence}">{_e(final.confidence.capitalize())}'
            f'<span class="meter {final.confidence}"><span></span><span></span><span></span></span></span>',
        ),
    ]
    if p.prob is not None:
        facts.append(("Probabilità", f"{p.prob * 100:.0f}%"))
    if report.geoclip_spread_km is not None:
        facts.append(("Dispersione GeoCLIP", f"~{report.geoclip_spread_km:,.0f} km".replace(",", ".")))
    facts.append(("Testi · oggetti", f"{len(report.ocr)} · {len(report.detections)}"))
    if report.forensics and report.forensics.hashes:
        facts.append(("Rif. evidenza", f'<span class="mono">{_e(report.forensics.hashes.sha256[:12])}</span>'))
    facts_html = "".join(f'<div class="fact"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in facts)
    address = f'<div class="addr">{_e(p.address)}</div>' if p.address else ""
    return (
        '<div class="verdict"><div class="verdict-top">'
        f'<div><div class="eyebrow">Verdetto</div><div class="place">{_e(p.label)}</div>{address}</div>'
        f'<div class="coords">{p.lat:.6f}, {p.lon:.6f}<small>{_e(to_dms(p.lat, p.lon))} · WGS84</small></div></div>'
        f'<p class="why">{_e(final.explanation)}</p><dl class="facts">{facts_html}</dl>{run_line(report.steps)}</div>'
    )


def map_legend(kinds: set[str] | None = None) -> str:
    names = [("final", "Verdetto"), ("exif", "GPS EXIF"), ("ai", "Ipotesi IA"), ("geoclip", "Candidati GeoCLIP")]
    keys = "".join(f'<span class="k {k}"><i></i><span>{label}</span></span>' for k, label in names if not kinds or k in kinds)
    return f'<div class="legend">{keys}<span class="datum">WGS84 · EPSG:4326</span></div>'


def formats_card(lat: float, lon: float) -> str:
    return section("Coordinate in tutti i formati", kv_table(all_formats(lat, lon).items(), mono=True))


def ai_card(ai: AiVerdict | None) -> str:
    if ai is None:
        return empty("Aggiungi una chiave Gemini nelle opzioni di analisi per far incrociare gli indizi all'IA.", "Arbitro IA non eseguito")
    if ai.error:
        return error_card(ai.error, "L'arbitro IA non ha risposto")
    stance = "con prove indipendenti" if ai.override else "senza prove sufficienti per un punto preciso"
    head = (
        f'<div class="eyebrow">{_e(ai.model)}</div><h3>{_e(ai.location_name or "Località non determinata")}</h3>'
        f'<p class="muted">{_e(ai.country or "Paese non determinato")} · confidenza {ai.confidence}% · {stance}</p>'
        f'<div class="conf-bar" role="img" aria-label="Confidenza {ai.confidence}%"><i style="width:{ai.confidence}%"></i></div>'
    )
    if ai.lat is not None and ai.lon is not None:
        head += f'<p class="mono">{ai.lat:.6f}, {ai.lon:.6f}</p>'
    parts = [f'<section class="sec">{head}</section>']
    if ai.clues:
        parts.append(section("Indizi osservati", "<ul>" + "".join(f"<li>{_e(c)}</li>" for c in ai.clues) + "</ul>"))
    if ai.reasoning:
        parts.append(section("Ragionamento", f"<p>{_e(ai.reasoning)}</p>"))
    if ai.alternatives:
        parts.append(section("Ipotesi alternative", "<ul>" + "".join(f"<li>{_e(c)}</li>" for c in ai.alternatives) + "</ul>"))
    if ai.sources:
        lis = "".join(
            f'<li><a href="{_e(s.get("url"))}" target="_blank" rel="noopener noreferrer">{_e(s.get("title"))}</a></li>' for s in ai.sources
        )
        parts.append(section("Fonti web consultate", f"<ul>{lis}</ul>"))
    return "".join(parts)


def _finding(text: str) -> str:
    low = text.lower()
    level = "bad" if text.startswith("⚠") or "discordanza" in low or "generato da ia" in low or "c2pa" in low else "warn"
    if "nessuna anomalia" in low:
        level = "ok"
    return notice(text.lstrip("⚠ ").strip(), level)


def metadata_card(meta: MetadataReport | None) -> str:
    if meta is None:
        return empty("Attiva il modulo Metadati nelle opzioni per leggere EXIF, GPS e XMP.", "Metadati non analizzati")
    gps = meta.gps
    rows = [
        ("File", meta.file_name),
        ("Formato", f"{meta.format} · {meta.mime}" if meta.mime else meta.format),
        ("Dimensioni", f"{meta.width} × {meta.height} px · {meta.mode}"),
        ("Peso", f"{meta.file_size / 1024:.1f} KB"),
        ("Fotocamera", meta.camera),
        ("Obiettivo", meta.lens),
        ("Software", meta.software),
        ("Data di scatto", " ".join(x for x in (meta.datetime_original, meta.timezone_offset) if x)),
        ("Qualità JPEG stimata", meta.jpeg_quality),
        ("Profilo colore", meta.icc_profile),
        ("Miniatura EXIF", "presente" if meta.has_thumbnail else ""),
    ]
    if gps:
        rows += [
            ("GPS", f"{gps.lat:.6f}, {gps.lon:.6f}"),
            ("Altitudine", f"{gps.altitude_m} m" if gps.altitude_m is not None else ""),
            ("Direzione di scatto", f"{gps.direction_deg:.0f}° ({gps.direction_ref or '?'})" if gps.direction_deg is not None else ""),
            ("Ora GPS (UTC)", gps.timestamp_utc),
        ]
    observations = [*meta.ai_markers, *meta.findings]
    obs = "".join(_finding(x) for x in observations) or notice("Nessuna osservazione particolare.", "ok")
    return section("Cosa rivela il file", kv_table(rows), "Riepilogo") + section("Osservazioni", obs)


def forensics_card(fz: ForensicsResult | None) -> str:
    if fz is None or fz.hashes is None:
        return empty("Attiva il modulo Forense nelle opzioni per calcolare hash ed Error Level Analysis.", "Analisi forense non eseguita")
    h = fz.hashes
    rows = [
        ("SHA-256", h.sha256),
        ("SHA-1", h.sha1),
        ("MD5", h.md5),
        ("pHash", h.phash),
        ("dHash", h.dhash),
        ("aHash", h.ahash),
        ("ELA medio (0-255)", f"{fz.ela_score:.2f}" if fz.ela_score is not None else ""),
    ]
    reading = (
        "<p class='muted'>ELA: le aree ritoccate tendono a risaltare con bordi netti rispetto al resto. "
        "Residuo di rumore: una grana diversa dal contesto può indicare parti incollate o generate. Sono indizi da verificare.</p>"
    )
    return section("Impronte del file", kv_table(rows, mono=True), "Catena di custodia") + section(
        "Indicatori", "".join(_finding(x) for x in fz.findings) + reading
    )


def ocr_card(texts: list[OcrText]) -> str:
    if not texts:
        return empty("Nessun testo leggibile, oppure il modulo OCR non è attivo.", "Testo")
    from ..geo.links import text_search_links

    rows = "".join(
        f'<div class="ocr-row"><span class="txt">{_e(t.text)}</span><span class="conf">{t.confidence:.0%}</span>'
        f"{links_html(text_search_links(t.text)[:4], grouped=False)}</div>"
        for t in texts
    )
    return section("Testo letto nell'immagine", f'<div class="ocr">{rows}</div>', f"{len(texts)} risultati")


def objects_card(counts: dict[str, int], relevant: set[str]) -> str:
    if not counts:
        return empty("Nessun oggetto riconosciuto, oppure il modulo YOLO non è attivo.", "Oggetti")
    pills = "".join(pill(f"{label} × {n}", "ok" if label in relevant else "idle") for label, n in counts.items())
    return section(
        "Oggetti riconosciuti",
        f'<div class="runline">{pills}</div><p class="muted">Pallino verde: classi utili alla geolocalizzazione (veicoli, segnaletica, infrastrutture).</p>',
    )


def steps_html(steps: list[StepStatus]) -> str:
    if not steps:
        return ""
    rows = [[s.name, STEP_LABEL.get(s.status, s.status), f"{s.duration_ms} ms", s.detail] for s in steps]
    return section("Registro di esecuzione", grid_table(["Modulo", "Esito", "Tempo", "Dettagli"], rows))


# ---------------------------------------------------------------------------
# Sun tools
# ---------------------------------------------------------------------------


def sun_path_svg(samples: list[SunPosition], marker: SunPosition | None = None, size: int = 300) -> str:
    """Polar sun-path diagram: azimuth around, elevation from horizon (edge) to zenith (centre)."""
    c = size / 2
    r = size / 2 - 24

    def xy(az: float, el: float) -> tuple[float, float]:
        rr = r * (90 - max(el, 0)) / 90
        a = math.radians(az)
        return c + rr * math.sin(a), c - rr * math.cos(a)

    grid = "".join(
        f'<circle cx="{c}" cy="{c}" r="{r * (90 - e) / 90:.1f}" fill="none" style="stroke:var(--go-rule)"/>'
        f'<text x="{c + 3}" y="{c - r * (90 - e) / 90 + 11:.1f}" font-size="9" style="fill:var(--go-ink-soft)">{e}°</text>'
        for e in (0, 30, 60)
    )
    grid += (
        f'<line x1="{c}" y1="{c - r}" x2="{c}" y2="{c + r}" style="stroke:var(--go-rule)"/>'
        f'<line x1="{c - r}" y1="{c}" x2="{c + r}" y2="{c}" style="stroke:var(--go-rule)"/>'
    )
    labels = "".join(
        f'<text x="{c + (r + 13) * math.sin(math.radians(a)):.1f}" y="{c - (r + 13) * math.cos(math.radians(a)) + 4:.1f}" '
        f'text-anchor="middle" font-size="11" font-weight="700" style="fill:var(--go-ink-soft)">{t}</text>'
        for a, t in ((0, "N"), (90, "E"), (180, "S"), (270, "O"))
    )
    above = [s for s in samples if s.elevation > 0]
    path = ""
    if above:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(s.azimuth, s.elevation) for s in above))
        path = f'<polyline points="{pts}" fill="none" style="stroke:var(--go-sun)" stroke-width="2.5" stroke-linecap="round"/>'
        for s in above:
            if s.when.minute == 0:
                x, y = xy(s.azimuth, s.elevation)
                path += (
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" style="fill:var(--go-sun)"/>'
                    f'<text x="{x + 5:.1f}" y="{y - 4:.1f}" font-size="9" style="fill:var(--go-ink-soft)">{s.when.hour}</text>'
                )
    mark = ""
    if marker and marker.elevation > 0:
        x, y = xy(marker.azimuth, marker.elevation)
        sx, sy = xy(marker.shadow_azimuth, 50)
        mark = (
            f'<line x1="{c}" y1="{c}" x2="{sx:.1f}" y2="{sy:.1f}" style="stroke:var(--go-ink)" stroke-width="3" stroke-dasharray="5 4"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" style="fill:var(--go-sun);stroke:var(--go-sheet)" stroke-width="2"/>'
        )
    return (
        f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:{size}px" role="img" '
        f'aria-label="Percorso del sole nel cielo durante la giornata">{grid}{labels}{path}{mark}</svg>'
    )


def fmt_time(value: datetime | None) -> str:
    return value.strftime("%H:%M") if isinstance(value, datetime) else "—"
