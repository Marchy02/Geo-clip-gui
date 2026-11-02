"""Visual identity: Gradio theme + CSS built on Gradio's own CSS variables,
so every custom card follows light/dark mode automatically."""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
    ).set(
        body_background_fill_dark="#0b1020",
        block_background_fill_dark="#111a2e",
        background_fill_secondary_dark="#0e1628",
        border_color_primary_dark="#1f2a44",
        button_primary_background_fill="linear-gradient(135deg, *primary_500, *primary_700)",
        button_primary_background_fill_hover="linear-gradient(135deg, *primary_400, *primary_600)",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="600",
    )


CSS = """
.gradio-container { max-width: 1480px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---------- header ---------- */
.go-header { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
  padding: 18px 22px; border-radius: 16px; border:1px solid var(--border-color-primary);
  background: radial-gradient(1200px 200px at 0% 0%, rgba(34,211,238,.18), transparent 60%),
              radial-gradient(800px 200px at 100% 0%, rgba(245,158,11,.12), transparent 60%),
              var(--block-background-fill); }
.go-brand { display:flex; align-items:center; gap:14px; }
.go-logo { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:24px;
  background: linear-gradient(135deg,#06b6d4,#0e7490); color:white; box-shadow:0 6px 24px rgba(6,182,212,.35); }
.go-title { font-size:1.45rem; font-weight:750; letter-spacing:-.02em; color: var(--body-text-color); line-height:1.1; }
.go-sub { color: var(--body-text-color-subdued); font-size:.88rem; }
.go-chips { display:flex; gap:6px; flex-wrap:wrap; }

/* ---------- chips / pills ---------- */
.chip { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:.78rem;
  border:1px solid var(--border-color-primary); background: var(--background-fill-secondary); color: var(--body-text-color); white-space:nowrap; }
.chip .dot { width:8px; height:8px; border-radius:50%; background:#94a3b8; }
.chip.ok .dot { background:#22c55e; } .chip.warn .dot { background:#f59e0b; } .chip.bad .dot { background:#ef4444; }

/* ---------- cards ---------- */
.card { border:1px solid var(--border-color-primary); border-radius:14px; padding:16px 18px; margin-bottom:10px;
  background: var(--block-background-fill); color: var(--body-text-color); font-size:.92rem; line-height:1.55; }
.card h3 { margin:0 0 10px; font-size:1rem; font-weight:650; }
.card h4 { margin:12px 0 6px; font-size:.85rem; text-transform:uppercase; letter-spacing:.06em; color: var(--body-text-color-subdued); }
.card ul { margin:4px 0; padding-left:18px; } .card li { margin:3px 0; }
.muted { color: var(--body-text-color-subdued); }
.empty { text-align:center; padding:36px 16px; color: var(--body-text-color-subdued); border:1px dashed var(--border-color-primary); border-radius:14px; }
.empty .ico { font-size:2rem; display:block; margin-bottom:6px; }

.verdict { position:relative; overflow:hidden; border-left:5px solid #06b6d4; }
.verdict.exif { border-left-color:#22c55e; } .verdict.ai { border-left-color:#f59e0b; } .verdict.geoclip { border-left-color:#06b6d4; }
.verdict .coords { font: 600 1.35rem/1.2 var(--font-mono); letter-spacing:-.01em; margin:4px 0; }
.verdict .place { font-size:1.05rem; font-weight:600; }
.verdict .why { margin-top:8px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:8px; margin-top:12px; }
.kpi { border:1px solid var(--border-color-primary); border-radius:10px; padding:8px 10px; background: var(--background-fill-secondary); }
.kpi b { display:block; font-size:1.02rem; line-height:1.3; } .kpi .lbl { display:block; margin-top:2px; font-size:.7rem; color: var(--body-text-color-subdued); text-transform:uppercase; letter-spacing:.05em; }

.conf-alta { color:#16a34a; } .conf-media { color:#d97706; } .conf-bassa { color:#dc2626; }
.bar { height:6px; border-radius:6px; background: var(--background-fill-secondary); overflow:hidden; }
.bar > i { display:block; height:100%; background: linear-gradient(90deg,#06b6d4,#22c55e); }

.kv, .prose .kv { width:100%; border-collapse:collapse !important; font-size:.86rem; border:none !important; background:transparent !important; margin:0 !important; }
.kv tr, .kv th, .kv td { border:none !important; background:transparent !important; }
.kv th { text-align:left; color: var(--body-text-color-subdued); font-weight:500; width:38%; padding:5px 8px 5px 0; vertical-align:top; }
.kv td { padding:5px 0; word-break:break-word; }
.kv tr + tr > th, .kv tr + tr > td { border-top:1px solid var(--border-color-primary) !important; }
.kv th, .kv td { padding-top:6px !important; padding-bottom:6px !important; }
.kv.wide th { width:auto; font-weight:600; }
.kv td { padding-left:0 !important; padding-right:0 !important; } .kv th { padding-left:0 !important; }
.mono, .kv code { font-family: var(--font-mono); font-size:.8rem; word-break:break-all; }

.links { display:flex; flex-wrap:wrap; gap:6px; }
.links a { text-decoration:none !important; padding:5px 10px; border-radius:9px; font-size:.82rem;
  border:1px solid var(--border-color-primary); background: var(--background-fill-secondary); color: var(--body-text-color) !important; }
.links a:hover { border-color:#06b6d4; color:#06b6d4 !important; }
.linkgroup + .linkgroup { margin-top:10px; }

.finding { padding:7px 10px; border-radius:9px; margin:5px 0; font-size:.86rem; background: var(--background-fill-secondary); border-left:3px solid #64748b; }
.finding.warn { border-left-color:#f59e0b; } .finding.bad { border-left-color:#ef4444; } .finding.good { border-left-color:#22c55e; }

.ocr-item { display:flex; justify-content:space-between; gap:10px; align-items:center; padding:7px 0; border-top:1px solid var(--border-color-primary); }
.ocr-item:first-child { border-top:none; }
.ocr-item code { font-size:.92rem; }

.steps { display:flex; flex-wrap:wrap; gap:6px; }
.map-frame { width:100%; height:460px; border:none; border-radius:14px; }
.go-primary button, button.go-primary { font-size:1.05rem !important; padding:12px !important; }
"""

HEAD = """
<meta name="description" content="GeoOSINT — coltellino svizzero OSINT per immagini">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛰️</text></svg>">
"""
