"""Visual identity of the dashboard.

Direction: a cartographer's chart table. Light mode is survey paper with a
slight green-grey bias, dark mode is a map room at night. The single accent
is aeronautical-chart magenta (the colour charts use for navigation aids);
coordinates and hashes are set in a mono face because they are read digit by
digit. Semantic colours (verified / warning / critical) are separate from the
accent. Only the verdict is a raised sheet; everything else sits on the page
separated by hairline rules.

Gradio toggles dark mode with a ``.dark`` class on the app root: tokens are
declared on ``:root`` and redefined under ``.dark``.
"""

from __future__ import annotations

import gradio as gr

# Layout: evidence column on the left (input -> options -> run), case file on the
# right (verdict -> map -> detail tabs). Summary always above detail.
TOKENS = """
:root {
  --go-paper: #f4f6f3;        /* survey paper */
  --go-sheet: #ffffff;        /* raised sheet: verdict */
  --go-ink: #18212b;          /* chart ink */
  --go-ink-soft: #5b6774;     /* annotations */
  --go-rule: #d9ded8;         /* hairline grid */
  --go-accent: #b0246f;       /* aeronautical magenta */
  --go-accent-soft: #f6e3ec;
  --go-contour: #8a5a2b;      /* contour brown: AI hypotheses */
  --go-water: #1f6f9f;        /* hydrography blue: GeoCLIP */
  --go-ok: #2f7d4f;
  --go-warn: #a8670a;
  --go-bad: #b42318;
  --go-sun: #c47a0c;          /* sun path */
  --go-display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --go-body: "Atkinson Hyperlegible", system-ui, -apple-system, "Segoe UI", sans-serif;
  --go-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  --go-radius: 6px;
}
.dark {
  --go-paper: #0f141a;
  --go-sheet: #161d26;
  --go-ink: #e4e9ee;
  --go-ink-soft: #95a2b0;
  --go-rule: #263140;
  --go-accent: #e0609f;
  --go-accent-soft: #3a1a2b;
  --go-contour: #d39a5f;
  --go-water: #5fb0e0;
  --go-ok: #5cc28a;
  --go-warn: #e0a440;
  --go-bad: #f07167;
  --go-sun: #f2b450;
}
"""

FONTS_HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@112,600;112,750'
    '&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)


def build_theme() -> gr.themes.Base:
    magenta = gr.themes.Color(
        c50="#fbeef4", c100="#f6dbe8", c200="#ecb5cf", c300="#e089b3", c400="#d15e97",
        c500="#b0246f", c600="#9a1f61", c700="#7f1950", c800="#64143f", c900="#4a0f2f", c950="#2e091d",
    )  # fmt: skip
    slate = gr.themes.Color(
        c50="#f4f6f3", c100="#e8ece7", c200="#d9ded8", c300="#b9c1c0", c400="#8b96a0",
        c500="#5b6774", c600="#434e5a", c700="#2c3642", c800="#1d2630", c900="#161d26", c950="#0f141a",
    )  # fmt: skip
    return gr.themes.Base(
        primary_hue=magenta,
        secondary_hue=magenta,
        neutral_hue=slate,
        radius_size=gr.themes.Size(xxs="2px", xs="3px", sm="4px", md="6px", lg="6px", xl="8px", xxl="10px"),
        font=["Atkinson Hyperlegible", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
    ).set(
        body_background_fill="#f4f6f3",
        body_background_fill_dark="#0f141a",
        body_text_color="#18212b",
        body_text_color_dark="#e4e9ee",
        body_text_color_subdued="#5b6774",
        body_text_color_subdued_dark="#95a2b0",
        block_background_fill="#ffffff",
        block_background_fill_dark="#161d26",
        background_fill_secondary="#eef1ed",
        background_fill_secondary_dark="#121921",
        border_color_primary="#d9ded8",
        border_color_primary_dark="#263140",
        block_border_width="1px",
        block_shadow="none",
        block_shadow_dark="none",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="#5b6774",
        block_label_text_color_dark="#95a2b0",
        block_label_text_weight="700",
        block_title_text_weight="700",
        input_background_fill="#ffffff",
        input_background_fill_dark="#0f141a",
        input_border_color="#c9d0c8",
        input_border_color_dark="#2c3846",
        input_border_color_focus="#b0246f",
        input_border_color_focus_dark="#e0609f",
        button_primary_background_fill="#b0246f",
        button_primary_background_fill_hover="#9a1f61",
        button_primary_background_fill_dark="#c13a82",
        button_primary_background_fill_hover_dark="#d15e97",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="#e8ece7",
        button_secondary_background_fill_hover_dark="#1d2630",
        button_secondary_border_color="#b9c1c0",
        button_secondary_border_color_dark="#2c3846",
        button_secondary_text_color="#18212b",
        button_secondary_text_color_dark="#e4e9ee",
        button_border_width="1px",
        checkbox_label_background_fill_selected="#f6e3ec",
        checkbox_label_background_fill_selected_dark="#3a1a2b",
        checkbox_background_color_selected="#b0246f",
        checkbox_background_color_selected_dark="#e0609f",
        slider_color="#b0246f",
        slider_color_dark="#e0609f",
        color_accent="#b0246f",
        color_accent_soft="#f6e3ec",
        color_accent_soft_dark="#3a1a2b",
        link_text_color="#b0246f",
        link_text_color_dark="#e0609f",
        table_even_background_fill="#ffffff",
        table_even_background_fill_dark="#161d26",
        table_odd_background_fill="#f7f9f6",
        table_odd_background_fill_dark="#131a22",
    )


CSS = (
    TOKENS
    + """
/* ---------- page ---------- */
.gradio-container { max-width: 1440px !important; width: 100%; margin: 0 auto !important; font-family: var(--go-body); color: var(--go-ink); }
.gradio-container, .gradio-container > .main, .gradio-container .wrap, .gradio-container .contain { min-width: 0 !important; }
footer { display: none !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { text-wrap: balance; }

/* ---------- masthead ---------- */
.go-mast { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 12px 24px;
  padding: 6px 2px 14px; border-bottom: 1px solid var(--go-rule); }
.go-mark { display: flex; align-items: center; gap: 12px; }
.go-mark svg { width: 38px; height: 38px; flex: none; color: var(--go-accent); }
.go-word { font-family: var(--go-display); font-stretch: 112%; font-weight: 750; font-size: 1.55rem; letter-spacing: -.01em; line-height: 1; color: var(--go-ink); }
.go-word small { font-family: var(--go-mono); font-size: .72rem; font-weight: 400; color: var(--go-ink-soft); letter-spacing: 0; margin-left: 6px; }
.go-tag { color: var(--go-ink-soft); font-size: .86rem; margin-top: 4px; }
.go-engines { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.go-engines .lbl { font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: var(--go-ink-soft); margin-right: 2px; }

/* ---------- pills (state, not decoration) ---------- */
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px; border-radius: 999px; font-size: .76rem; line-height: 1.6;
  border: 1px solid var(--go-rule); color: var(--go-ink); background: var(--go-sheet); white-space: nowrap; }
.pill i { width: 7px; height: 7px; border-radius: 50%; background: var(--go-ink-soft); }
.pill.ok i { background: var(--go-ok); } .pill.warn i { background: var(--go-warn); } .pill.bad i { background: var(--go-bad); }
.pill.idle i { background: transparent; border: 1.5px solid var(--go-ink-soft); }

/* ---------- sections ---------- */
.eyebrow { font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--go-ink-soft); margin: 0 0 6px; }
.sec { padding: 4px 0 2px; color: var(--go-ink); font-size: .93rem; line-height: 1.55; }
.sec + .sec { border-top: 1px solid var(--go-rule); margin-top: 14px; padding-top: 14px; }
.sec h3 { font-family: var(--go-display); font-stretch: 112%; font-weight: 600; font-size: 1.02rem; margin: 0 0 8px; color: var(--go-ink); }
.sec p { margin: 6px 0; max-width: 72ch; }
.muted { color: var(--go-ink-soft); }
.mono { font-family: var(--go-mono); font-variant-numeric: tabular-nums; }

/* ---------- empty / error states ---------- */
.state { display: grid; gap: 4px; justify-items: start; padding: 20px; border: 1px dashed var(--go-rule);
  border-radius: var(--go-radius); color: var(--go-ink-soft); background: transparent; font-size: .9rem; }
.state b { color: var(--go-ink); font-weight: 700; }
.state span { color: var(--go-ink-soft); }
.state.error { border-style: solid; border-color: var(--go-bad); }
.state.error b { color: var(--go-bad); }

/* ---------- verdict: the one raised sheet ---------- */
.verdict { display: grid; gap: 14px; padding: 18px 20px 14px; background: var(--go-sheet); border: 1px solid var(--go-rule); border-radius: var(--go-radius); }
.verdict-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px 20px; }
.verdict-top > div { min-width: 0; }
.verdict .place { font-family: var(--go-display); font-stretch: 112%; font-weight: 750; font-size: 1.4rem; line-height: 1.15; color: var(--go-ink); text-wrap: balance; }
.verdict .addr { color: var(--go-ink-soft); font-size: .88rem; margin-top: 3px; }
.verdict .coords { font-family: var(--go-mono); font-variant-numeric: tabular-nums; font-size: 1.12rem; font-weight: 600; color: var(--go-accent); text-align: right; }
.verdict .coords small { display: block; font-size: .78rem; font-weight: 400; color: var(--go-ink-soft); margin-top: 2px; }
.verdict .why { max-width: 80ch; font-size: .93rem; color: var(--go-ink); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin: 0; border-top: 1px solid var(--go-rule); }
.fact { padding: 10px 12px 2px 0; margin: 0; }
.fact + .fact { padding-left: 12px; border-left: 1px solid var(--go-rule); }
.fact dt { font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: var(--go-ink-soft); }
.fact dd { margin: 2px 0 0; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--go-ink); }
.lvl-alta { color: var(--go-ok) !important; } .lvl-media { color: var(--go-warn) !important; } .lvl-bassa { color: var(--go-bad) !important; }
.meter { display: flex; gap: 3px; margin-top: 5px; max-width: 90px; } .meter span { flex: 1; height: 4px; border-radius: 2px; background: var(--go-rule); }
.meter.alta span:nth-child(-n+3) { background: var(--go-ok); } .meter.media span:nth-child(-n+2) { background: var(--go-warn); }
.meter.bassa span:nth-child(1) { background: var(--go-bad); }
.runline { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.runline .lbl { font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: var(--go-ink-soft); margin-right: 2px; }

/* ---------- map legend ---------- */
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; font-size: .8rem; color: var(--go-ink-soft); padding: 2px 2px 8px; }
.legend .datum { margin-left: auto; font-family: var(--go-mono); font-size: .74rem; color: var(--go-ink-soft); }
.legend .k { display: inline-flex; align-items: center; gap: 6px; }
.legend .k i { width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--k); }
.legend .k.final i { background: var(--k); }
.k.final { --k: var(--go-accent); } .k.exif { --k: var(--go-ok); } .k.ai { --k: var(--go-contour); } .k.geoclip { --k: var(--go-water); }
.legend .k span { color: var(--go-ink-soft); }
.map-frame { display: block; width: 100%; border: 1px solid var(--go-rule); border-radius: var(--go-radius); background: var(--go-sheet); }

/* ---------- key/value tables ---------- */
.kv-wrap { overflow-x: auto; }
.kv, .prose .kv { width: 100%; border-collapse: collapse !important; border: none !important; margin: 0 !important; background: transparent !important; font-size: .88rem; }
.kv tr, .kv th, .kv td { border: none !important; background: transparent !important; }
.kv th, .kv td { padding: 6px 12px 6px 0 !important; text-align: left; vertical-align: top; color: var(--go-ink); }
.kv th { color: var(--go-ink-soft) !important; font-weight: 400; width: 34%; }
.kv td { word-break: break-word; font-variant-numeric: tabular-nums; }
.kv tr + tr > th, .kv tr + tr > td { border-top: 1px solid var(--go-rule) !important; }
.kv code, .kv .mono { font-family: var(--go-mono); font-size: .82rem; word-break: break-all; background: none; padding: 0; color: var(--go-ink); }
.kv.grid th { width: auto; font-weight: 700; font-size: .7rem; letter-spacing: .06em; text-transform: uppercase; }

/* ---------- findings: severity stripe encodes state ---------- */
.finding { display: grid; grid-template-columns: 4px minmax(0, 1fr); gap: 10px; padding: 5px 0; font-size: .9rem; color: var(--go-ink); }
.finding::before { content: ""; border-radius: 2px; background: var(--go-ink-soft); }
.finding.warn::before { background: var(--go-warn); } .finding.bad::before { background: var(--go-bad); } .finding.ok::before { background: var(--go-ok); }

/* ---------- links ---------- */
.linkgroups { display: grid; gap: 18px 24px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.linkgroup ul { list-style: none !important; margin: 0 !important; padding: 0 !important; }
.linkgroup li { margin: 0 !important; padding: 0 !important; }
.linkgroup a { display: grid; grid-template-columns: minmax(0, 1fr) auto; column-gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--go-rule);
  text-decoration: none !important; font-size: .9rem; }
.linkgroup a, .linkgroup a .nm, .linkgroup a small { text-align: left !important; justify-self: start; }
.linkgroup a .nm { color: var(--go-ink); }
.linkgroup a::after { content: "↗"; color: var(--go-ink-soft); font-size: .8rem; grid-column: 2; grid-row: 1; }
.linkgroup a small { grid-column: 1 / -1; color: var(--go-ink-soft); font-size: .76rem; line-height: 1.3; }
.linkgroup a:hover .nm, .linkgroup a:focus-visible .nm, .linkgroup a:hover::after { color: var(--go-accent); }
.inline-links { display: inline-flex; flex-wrap: wrap; gap: 4px 14px; font-size: .82rem; }
.inline-links a { color: var(--go-accent) !important; text-decoration: none !important; }
.inline-links a:hover { text-decoration: underline !important; }

/* ---------- OCR ---------- */
.ocr-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 16px; align-items: baseline; padding: 9px 0; border-top: 1px solid var(--go-rule); }
.ocr-row:first-child { border-top: none; }
.ocr-row .txt { font-family: var(--go-mono); font-size: .95rem; word-break: break-word; color: var(--go-ink); }
.ocr-row .conf { font-size: .78rem; color: var(--go-ink-soft); font-variant-numeric: tabular-nums; }
.ocr-row .inline-links { grid-column: 1 / -1; }

/* ---------- AI ---------- */
.conf-bar { height: 6px; border-radius: 3px; background: var(--go-rule); overflow: hidden; max-width: 320px; margin: 6px 0 2px; }
.conf-bar > i { display: block; height: 100%; background: var(--go-contour); }
.sec ul { margin: 4px 0; padding-left: 18px; } .sec li { margin: 3px 0; }

/* ---------- sun diagram ---------- */
.sun-figure { display: grid; justify-items: center; gap: 6px; }
.sun-figure figcaption { font-size: .8rem; color: var(--go-ink-soft); text-align: center; }

/* ---------- controls ---------- */
button.go-run { font-family: var(--go-display) !important; font-stretch: 112%; font-weight: 600 !important; letter-spacing: .01em; min-height: 46px; font-size: 1rem !important; }
.go-hint { font-size: .82rem; color: var(--go-ink-soft); }
.go-intro { color: var(--go-ink-soft); max-width: 80ch; }
:focus-visible { outline: 2px solid var(--go-accent) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }

@media (max-width: 640px) {
  .gradio-container .row { flex-wrap: wrap !important; }
  .gradio-container .row > .column { min-width: 100% !important; flex: 1 1 100% !important; }
  .gradio-container .table-wrap { max-width: 100%; overflow-x: auto; }
  .fact + .fact { border-left: none; padding-left: 0; }
  .legend .datum { margin-left: 0; }
  .verdict .place { font-size: 1.2rem; }
  .verdict .coords { text-align: left; }
}
"""
)

HEAD = (
    FONTS_HEAD
    + '<meta name="description" content="GeoOSINT: geolocalizzazione e verifica di immagini">'
    + "<link rel=\"icon\" href=\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    + "<circle cx='16' cy='16' r='12' fill='none' stroke='%23b0246f' stroke-width='3'/>"
    + "<path d='M16 1v9M16 22v9M1 16h9M22 16h9' stroke='%23b0246f' stroke-width='3'/></svg>\">"
)
