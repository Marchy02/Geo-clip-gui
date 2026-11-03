"""Report export: JSON (machine), Markdown (notes/wiki), HTML (self-contained dossier)."""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..geo.coords import all_formats
from ..geo.links import map_links, reverse_image_links, text_search_links
from ..models import AnalysisReport

FORMATS = ("json", "md", "html")


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _img_b64(img: Image.Image | str | None, max_side: int = 1200) -> str | None:
    if img is None:
        return None
    try:
        im = Image.open(img) if isinstance(img, (str, Path)) else img
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _safe_json(markers: list[dict]) -> str:
    """JSON for an inline <script>: labels are HTML-escaped (Leaflet tooltips render HTML)
    and '</' is neutralised so the payload cannot close the script tag."""
    clean = [{**m, "label": html.escape(str(m.get("label", "")))} for m in markers]
    return json.dumps(clean, ensure_ascii=False).replace("</", "<\\/")


def to_json(report: AnalysisReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def to_markdown(report: AnalysisReport) -> str:
    md: list[str] = [f"# Report OSINT — {Path(report.image_path).name}", ""]
    md.append(f"*Generato il {report.created_at} con GeoOSINT {report.tool_version}*")
    md.append("")
    if report.final:
        f = report.final
        md += [
            "## Verdetto",
            "",
            f"- **Posizione:** {f.point.lat:.6f}, {f.point.lon:.6f}" + (f" — {f.point.label}" if f.point.label else ""),
            f"- **Fonte:** {f.source.label}",
            f"- **Affidabilità:** {f.confidence}",
        ]
        if f.point.address:
            md.append(f"- **Indirizzo:** {f.point.address}")
        md += [f"- **Motivazione:** {f.explanation}", ""]
        md += ["### Formati coordinate", ""] + [f"- {k}: `{v}`" for k, v in all_formats(f.point.lat, f.point.lon).items()]
        md += ["", "### Link di verifica", ""]
        md += [f"- [{link.name}]({link.url})" for link in map_links(f.point.lat, f.point.lon)[:12]]
        md.append("")
    if report.ai:
        a = report.ai
        md += ["## Arbitro IA", ""]
        if a.error:
            md += [f"> Errore: {a.error}", ""]
        else:
            md += [
                f"- **Luogo:** {a.location_name or 'n/d'} ({a.country or 'paese n/d'})",
                f"- **Confidenza:** {a.confidence}%",
                f"- **Override:** {'sì' if a.override else 'no'}",
                "",
            ]
            if a.clues:
                md += ["**Indizi:**", ""] + [f"- {c}" for c in a.clues] + [""]
            if a.reasoning:
                md += ["**Ragionamento:** " + a.reasoning, ""]
            if a.alternatives:
                md += ["**Ipotesi alternative:** " + "; ".join(a.alternatives), ""]
            if a.sources:
                md += ["**Fonti web:**", ""] + [f"- [{s['title']}]({s['url']})" for s in a.sources] + [""]
    if report.geoclip:
        md += ["## GeoCLIP — candidati", "", "| # | Lat | Lon | Prob. | Zona |", "|---|---|---|---|---|"]
        md += [f"| {i + 1} | {g.lat:.5f} | {g.lon:.5f} | {(g.prob or 0) * 100:.2f}% | {g.address} |" for i, g in enumerate(report.geoclip)]
        if report.geoclip_spread_km is not None:
            md.append(f"\nDispersione media dei candidati: ~{report.geoclip_spread_km:.0f} km")
        md.append("")
    if report.ocr:
        md += ["## Testo (OCR)", ""] + [f"- `{t.text}` ({t.confidence:.0%})" for t in report.ocr] + [""]
    if report.detections:
        counts: dict[str, int] = {}
        for d in report.detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        md += ["## Oggetti (YOLO)", ""] + [f"- {k}: {v}" for k, v in counts.items()] + [""]
    m = report.metadata
    if m:
        md += ["## Metadati", "", f"- File: `{m.file_name}` — {m.format} {m.width}×{m.height}, {m.file_size:,} byte".replace(",", ".")]
        for label, value in (
            ("Fotocamera", m.camera),
            ("Obiettivo", m.lens),
            ("Software", m.software),
            ("Data scatto", m.datetime_original),
            ("Qualità JPEG stimata", m.jpeg_quality),
        ):
            if value:
                md.append(f"- {label}: {value}")
        if m.gps:
            md.append(f"- GPS: {m.gps.lat:.6f}, {m.gps.lon:.6f}" + (f", alt. {m.gps.altitude_m} m" if m.gps.altitude_m is not None else ""))
        if m.findings or m.ai_markers:
            md += ["", "**Osservazioni:**", ""] + [f"- {x}" for x in [*m.ai_markers, *m.findings]]
        md.append("")
    fz = report.forensics
    if fz and fz.hashes:
        h = fz.hashes
        md += [
            "## Forense",
            "",
            f"- MD5 `{h.md5}`",
            f"- SHA-1 `{h.sha1}`",
            f"- SHA-256 `{h.sha256}`",
            f"- pHash `{h.phash}` · dHash `{h.dhash}` · aHash `{h.ahash}`",
        ]
        if fz.ela_score is not None:
            md.append(f"- ELA medio: {fz.ela_score:.2f}")
        md += [f"- {x}" for x in fz.findings] + [""]
    if report.steps:
        md += ["## Esecuzione", "", "| Step | Esito | Tempo | Dettagli |", "|---|---|---|---|"]
        md += [f"| {s.name} | {s.status} | {s.duration_ms} ms | {s.detail.replace('|', '/')} |" for s in report.steps]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_CSS = """
:root{--bg:#f6f7fb;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#0891b2;--warn:#b45309;--bad:#b91c1c;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#0b1020;--card:#121a2e;--ink:#e2e8f0;--muted:#94a3b8;--line:#1e293b;--accent:#22d3ee;--warn:#f59e0b;--bad:#f87171;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.1rem;margin:0 0 12px}
.sub{color:var(--muted);margin-bottom:24px}.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px;overflow:hidden}
.verdict{border-left:5px solid var(--accent)}.big{font-size:1.25rem;font-weight:650}.muted{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:.9rem}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;word-break:break-word}
th{color:var(--muted);font-weight:600}code{font:12px ui-monospace,Menlo,Consolas,monospace;word-break:break-all}
img{max-width:100%;border-radius:10px;display:block}#map{height:380px;border-radius:12px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;border:1px solid var(--line);font-size:.8rem;margin:2px}
.alta{color:var(--ok)}.media{color:var(--warn)}.bassa{color:var(--bad)}ul{padding-left:20px;margin:6px 0}a{color:var(--accent)}
.links a{display:inline-block;margin:3px 6px 3px 0}
"""


def _table(rows: list[tuple[str, str]]) -> str:
    return "<table>" + "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows if v not in (None, "")) + "</table>"


def to_html(report: AnalysisReport) -> str:
    name = Path(report.image_path).name
    parts: list[str] = []
    markers: list[dict] = []

    if report.final:
        f = report.final
        markers.append({"lat": f.point.lat, "lon": f.point.lon, "label": f"VERDETTO: {f.point.label}", "color": "#ef4444"})
        links = "".join(
            f'<a href="{_e(link.url)}" target="_blank" rel="noopener">{_e(link.name)}</a>' for link in map_links(f.point.lat, f.point.lon)
        )
        parts.append(
            f'<div class="card verdict"><h2>Verdetto</h2><div class="big">{f.point.lat:.6f}, {f.point.lon:.6f}</div>'
            f"<div>{_e(f.point.label)}{(' — ' + _e(f.point.address)) if f.point.address else ''}</div>"
            f'<p><span class="pill">Fonte: {_e(f.source.label)}</span> <span class="pill {f.confidence}">Affidabilità {_e(f.confidence)}</span></p>'
            f"<p>{_e(f.explanation)}</p>{_table(list(all_formats(f.point.lat, f.point.lon).items()))}"
            f'<p class="links">{links}</p></div>'
        )
    for i, g in enumerate(report.geoclip):
        markers.append(
            {"lat": g.lat, "lon": g.lon, "label": f"GeoCLIP #{i + 1} {(g.prob or 0) * 100:.1f}% {g.address}", "color": "#22d3ee"}
        )
    if report.metadata and report.metadata.gps:
        markers.append({"lat": report.metadata.gps.lat, "lon": report.metadata.gps.lon, "label": "GPS EXIF", "color": "#22c55e"})
    if report.ai and report.ai.lat is not None and report.ai.lon is not None:
        markers.append({"lat": report.ai.lat, "lon": report.ai.lon, "label": f"IA: {report.ai.location_name}", "color": "#f59e0b"})

    if markers:
        parts.append('<div class="card"><h2>Mappa</h2><div id="map"></div></div>')

    img_src = _img_b64(report.image_path)
    visuals = []
    if img_src:
        visuals.append(f'<div class="card"><h2>Immagine</h2><img src="{img_src}" alt="immagine analizzata"></div>')
    if report.detections_image is not None:
        visuals.append(f'<div class="card"><h2>Oggetti rilevati</h2><img src="{_img_b64(report.detections_image)}" alt="YOLO"></div>')
    if report.forensics and report.forensics.ela_image is not None:
        visuals.append(f'<div class="card"><h2>Error Level Analysis</h2><img src="{_img_b64(report.forensics.ela_image)}" alt="ELA"></div>')
    if visuals:
        parts.append('<div class="grid">' + "".join(visuals) + "</div>")

    a = report.ai
    if a:
        if a.error:
            body = f'<p class="bassa">{_e(a.error)}</p>'
        else:
            body = (
                f'<div class="big">{_e(a.location_name or "n/d")}</div><div class="muted">{_e(a.country)} · confidenza {a.confidence}% · override {"sì" if a.override else "no"} · {_e(a.model)}</div>'
                + (("<h3>Indizi</h3><ul>" + "".join(f"<li>{_e(c)}</li>" for c in a.clues) + "</ul>") if a.clues else "")
                + (f"<h3>Ragionamento</h3><p>{_e(a.reasoning)}</p>" if a.reasoning else "")
                + (("<h3>Alternative</h3><ul>" + "".join(f"<li>{_e(c)}</li>" for c in a.alternatives) + "</ul>") if a.alternatives else "")
                + (
                    (
                        "<h3>Fonti web</h3><ul>"
                        + "".join(
                            f'<li><a href="{_e(s["url"])}" target="_blank" rel="noopener">{_e(s["title"])}</a></li>' for s in a.sources
                        )
                        + "</ul>"
                    )
                    if a.sources
                    else ""
                )
            )
        parts.append(f'<div class="card"><h2>Arbitro IA</h2>{body}</div>')

    side: list[str] = []
    if report.geoclip:
        rows = "".join(
            f"<tr><td>{i + 1}</td><td>{g.lat:.5f}, {g.lon:.5f}</td><td>{(g.prob or 0) * 100:.2f}%</td><td>{_e(g.address)}</td></tr>"
            for i, g in enumerate(report.geoclip)
        )
        spread = f'<p class="muted">Dispersione ~{report.geoclip_spread_km:.0f} km</p>' if report.geoclip_spread_km is not None else ""
        side.append(
            f'<div class="card"><h2>GeoCLIP</h2><table><tr><th>#</th><th>Coordinate</th><th>Prob.</th><th>Zona</th></tr>{rows}</table>{spread}</div>'
        )
    if report.ocr:
        items = "".join(
            f'<li><code>{_e(t.text)}</code> <span class="muted">{t.confidence:.0%}</span> '
            + " ".join(
                f'<a href="{_e(link.url)}" target="_blank" rel="noopener">{_e(link.name)}</a>' for link in text_search_links(t.text)[:2]
            )
            + "</li>"
            for t in report.ocr
        )
        side.append(f'<div class="card"><h2>Testo (OCR)</h2><ul>{items}</ul></div>')
    if side:
        parts.append('<div class="grid">' + "".join(side) + "</div>")

    m = report.metadata
    if m:
        info = _table(
            [
                ("File", m.file_name),
                ("Formato", f"{m.format} · {m.mime}"),
                ("Dimensioni", f"{m.width}×{m.height} px"),
                ("Peso", f"{m.file_size:,} byte".replace(",", ".")),
                ("Fotocamera", m.camera),
                ("Obiettivo", m.lens),
                ("Software", m.software),
                ("Data scatto", f"{m.datetime_original or ''} {m.timezone_offset or ''}".strip()),
                ("Qualità JPEG", m.jpeg_quality or ""),
                ("Profilo ICC", m.icc_profile or ""),
                ("GPS", f"{m.gps.lat:.6f}, {m.gps.lon:.6f}" if m.gps else ""),
            ]
        )
        obs = [*m.ai_markers, *m.findings]
        obs_html = ("<h3>Osservazioni</h3><ul>" + "".join(f"<li>{_e(x)}</li>" for x in obs) + "</ul>") if obs else ""
        exif_html = (
            ("<details><summary>Tutti i tag EXIF (" + str(len(m.exif)) + ")</summary>" + _table(list(m.exif.items())) + "</details>")
            if m.exif
            else ""
        )
        xmp_html = (
            ("<details><summary>XMP (" + str(len(m.xmp)) + ")</summary>" + _table(list(m.xmp.items())) + "</details>") if m.xmp else ""
        )
        parts.append(f'<div class="card"><h2>Metadati</h2>{info}{obs_html}{exif_html}{xmp_html}</div>')

    fz = report.forensics
    if fz and fz.hashes:
        h = fz.hashes
        rows = [
            ("MD5", h.md5),
            ("SHA-1", h.sha1),
            ("SHA-256", h.sha256),
            ("pHash", h.phash),
            ("dHash", h.dhash),
            ("aHash", h.ahash),
            ("ELA medio", f"{fz.ela_score:.2f}" if fz.ela_score is not None else ""),
        ]
        table = "<table>" + "".join(f"<tr><th>{k}</th><td><code>{_e(v)}</code></td></tr>" for k, v in rows if v) + "</table>"
        parts.append(f'<div class="card"><h2>Forense</h2>{table}<ul>' + "".join(f"<li>{_e(x)}</li>" for x in fz.findings) + "</ul></div>")

    rev = "".join(f'<a href="{_e(link.url)}" target="_blank" rel="noopener">{_e(link.name)}</a>' for link in reverse_image_links())
    parts.append(f'<div class="card"><h2>Ricerca inversa</h2><p class="links">{rev}</p></div>')

    if report.steps:
        rows = "".join(
            f"<tr><td>{_e(s.name)}</td><td>{_e(s.status)}</td><td>{s.duration_ms} ms</td><td>{_e(s.detail)}</td></tr>" for s in report.steps
        )
        parts.append(
            f'<div class="card"><h2>Esecuzione</h2><table><tr><th>Step</th><th>Esito</th><th>Tempo</th><th>Dettagli</th></tr>{rows}</table></div>'
        )

    map_js = ""
    if markers:
        map_js = (
            '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
            '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>'
            f"const M={_safe_json(markers)};"
            "const map=L.map('map');L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(map);"
            "const g=L.featureGroup(M.map(m=>L.circleMarker([m.lat,m.lon],{radius:8,color:m.color,fillOpacity:.8}).bindTooltip(m.label))).addTo(map);"
            "map.fitBounds(g.getBounds().pad(0.3),{maxZoom:14});</script>"
        )

    return (
        "<!doctype html><html lang='it'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Report OSINT · {_e(name)}</title><style>{_HTML_CSS}</style></head><body><main>"
        f"<h1>Report OSINT</h1><div class='sub'>{_e(name)} · {_e(report.created_at)} · GeoOSINT {_e(report.tool_version)}</div>"
        + "".join(parts)
        + f"</main>{map_js}</body></html>"
    )


def save_reports(report: AnalysisReport, out_dir: str | Path, formats: tuple[str, ...] = FORMATS) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(report.image_path).stem[:40] or "immagine"
    digest = report.forensics.hashes.sha256[:8] if report.forensics and report.forensics.hashes else "nohash"
    base = f"{stem}_{datetime.now():%Y%m%d-%H%M%S}_{digest}"
    writers = {"json": to_json, "md": to_markdown, "html": to_html}
    paths: dict[str, Path] = {}
    for fmt in formats:
        if fmt not in writers:
            raise ValueError(f"Formato report sconosciuto: {fmt}")
        path = out / f"{base}.{fmt}"
        path.write_text(writers[fmt](report), encoding="utf-8")
        paths[fmt] = path
    return paths
