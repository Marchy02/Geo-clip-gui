"""Gradio dashboard.

Information architecture
- Analisi: evidence column (input, options, run) | case file (verdict, map, detail tabs).
  The verdict sits above the map, which sits above the detail: summary before detail.
- Coordinate, Sole e ombre, Confronto, Batch, Privacy: single-purpose tools that open
  in a working state with an example already computed.
- Guida: workflow and engine status.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import gradio as gr

from ..config import get_settings
from ..engines.geoclip_engine import presets
from . import handlers as h
from . import settings_handlers as sh
from .settings_tab import build_settings_tab
from .theme import CSS, HEAD, build_theme

EXAMPLE_IMAGE = Path(__file__).resolve().parents[2] / "geoclip" / "images" / "Kauai.png"
EXAMPLE_PLACE = "41.890210, 12.492231"  # Colosseo, Roma

GUIDE = """
### Flusso di lavoro
1. **Analizza.** Carica la foto e premi *Analizza immagine*. Il verdetto sceglie la fonte più affidabile
   nell'ordine: GPS nei metadati, arbitro IA con prove indipendenti, stima GeoCLIP. Leggi sempre la motivazione.
2. **Verifica sul posto.** Dalla scheda *Geolocalizzazione* apri Street View, Mapillary o il satellite e confronta
   edifici, cartelli, pali, vegetazione e profilo dell'orizzonte.
3. **Allarga la ricerca.** Cerca i testi letti dall'OCR e usa la ricerca inversa per trovare la copia più vecchia
   della foto, spesso con la didascalia originale.
4. **Data e ora.** Nella scheda *Sole e ombre* ricavi l'orario dalla lunghezza e dalla direzione di un'ombra.
5. **Documenta.** Ogni analisi produce un report HTML, JSON e Markdown con l'hash SHA-256 del file analizzato.

### Consigli
* Le foto passate da social e chat perdono i metadati: la ricerca inversa può portarti all'originale.
* Candidati GeoCLIP molto dispersi indicano una scena poco caratteristica: dai più peso agli altri indizi.
* Nelle *note per l'IA* scrivi ciò che già sai, per esempio "video pubblicato a febbraio, si parla di una città costiera".
* Per foto da Ucraina e Russia scegli il modello GeoCLIP regionale.
* Usa lo strumento solo per indagini legittime, nel rispetto della legge e della privacy delle persone.
"""


def build_app() -> gr.Blocks:
    s = get_settings()
    preset_choices = [(desc, key) for key, desc in presets().items()]
    lang_labels = list(h.OCR_LANGS)
    default_langs = [label for label, code in h.OCR_LANGS.items() if code in s.ocr_languages] or ["Inglese"]
    all_modules = list(h.MODULES)
    active = s.provider()
    if active and active.is_ready():
        ai_hint = f"Arbitro IA attivo: {active.name} ({active.model})."
    elif active:
        ai_hint = f"Arbitro IA {active.name}: {active.problem()}. Completa la configurazione in Impostazioni."
    else:
        ai_hint = "Nessun provider IA configurato: l'analisi usa solo i motori locali. Aggiungine uno in Impostazioni."

    with gr.Blocks(title="GeoOSINT") as demo:
        header = gr.HTML(h.header_html())

        with gr.Tabs():
            # ------------------------------------------------------------------
            with gr.Tab("Analisi", id="analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        gr.HTML('<div class="eyebrow">Evidenza</div>')
                        image = gr.Image(
                            type="filepath",
                            label="Foto da analizzare",
                            sources=["upload", "clipboard"],
                            height=300,
                        )
                        if EXAMPLE_IMAGE.exists():
                            gr.Examples([[str(EXAMPLE_IMAGE)]], inputs=image, label="Esempio pronto (Kauai, Hawaii)")
                        with gr.Row(equal_height=True):
                            url = gr.Textbox(
                                placeholder="oppure incolla l'URL di un'immagine",
                                show_label=False,
                                scale=5,
                                container=False,
                                lines=1,
                                max_lines=1,
                            )
                            fetch_btn = gr.Button("Scarica", scale=1, size="sm", min_width=80)
                        fetch_msg = gr.HTML()
                        with gr.Accordion("Opzioni di analisi", open=False):
                            modules = gr.CheckboxGroup(all_modules, value=all_modules, label="Moduli da eseguire")
                            with gr.Row():
                                preset = gr.Dropdown(preset_choices, value=s.geoclip_preset, label="Modello GeoCLIP", scale=3)
                                top_k = gr.Slider(1, 20, value=s.geoclip_top_k, step=1, label="Candidati", scale=2)
                            ocr_langs = gr.Dropdown(lang_labels, value=default_langs, multiselect=True, label="Lingue da leggere (OCR)")
                            with gr.Row():
                                ai_provider = gr.Dropdown(
                                    sh.provider_choices(), value=active.id if active else None, label="Provider IA", scale=3
                                )
                                use_search = gr.Checkbox(
                                    value=bool(active and active.web_search),
                                    label="Ricerca web",
                                    scale=2,
                                    info="se il provider la supporta",
                                )
                            with gr.Row():
                                model = gr.Textbox(label="Modello", placeholder="vuoto = quello del provider", scale=3)
                                api_key = gr.Textbox(
                                    label="Chiave per questa sessione",
                                    type="password",
                                    placeholder="facoltativa, non viene salvata",
                                    scale=3,
                                )
                            notes = gr.Textbox(
                                label="Note per l'IA",
                                placeholder="Ciò che sai già, es. pubblicata su Telegram il 3 marzo, si parla di una città costiera",
                                lines=2,
                            )
                        with gr.Row():
                            run_btn = gr.Button("Analizza immagine", variant="primary", scale=3, elem_classes="go-run")
                            clear_btn = gr.ClearButton(value="Ricomincia", scale=1, min_width=100)
                        gr.HTML(f'<p class="go-hint">{ai_hint}</p>')

                    with gr.Column(scale=7, min_width=420):
                        first = h.blank_outputs()
                        verdict = gr.HTML(first[0])
                        map_html = gr.HTML(first[1])
                        with gr.Tabs():
                            with gr.Tab("Geolocalizzazione"):
                                geo_df = gr.Dataframe(
                                    headers=["#", "Lat", "Lon", "Probabilità", "Zona"],
                                    label="Candidati GeoCLIP",
                                    interactive=False,
                                    wrap=True,
                                )
                                geo_html = gr.HTML()
                            with gr.Tab("Arbitro IA"):
                                ai_html = gr.HTML()
                            with gr.Tab("Metadati"):
                                meta_html = gr.HTML()
                                exif_df = gr.Dataframe(
                                    headers=["Tag", "Valore"], label="Tutti i tag", interactive=False, wrap=True, max_height=420
                                )
                            with gr.Tab("Forense"):
                                with gr.Row():
                                    ela_img = gr.Image(label="Error Level Analysis", interactive=False, height=300)
                                    noise_img = gr.Image(label="Residuo di rumore", interactive=False, height=300)
                                forensic_html = gr.HTML()
                            with gr.Tab("Testo e oggetti"):
                                ocr_html = gr.HTML()
                                yolo_img = gr.Image(label="Oggetti riconosciuti", interactive=False, height=360)
                                yolo_html = gr.HTML()
                            with gr.Tab("Ricerca inversa"):
                                reverse_html = gr.HTML()
                            with gr.Tab("Report"):
                                gr.HTML(
                                    '<p class="go-intro">Ogni analisi salva tre report: HTML da consultare o allegare, '
                                    "JSON con tutti i dati, Markdown per le note del caso.</p>"
                                )
                                report_files = gr.File(label="Scarica i report", file_count="multiple", interactive=False)
                                report_json = gr.JSON(label="Dati completi", open=False)
                            with gr.Tab("Registro"):
                                steps = gr.HTML()

                outputs = [
                    verdict, map_html, geo_df, geo_html, ai_html, meta_html, exif_df, ela_img, noise_img,
                    forensic_html, ocr_html, yolo_img, yolo_html, reverse_html, report_files, steps, report_json,
                ]  # fmt: skip

                def _run(img, mods, pre, k, langs, key, mdl, search, nts, img_url, prov, progress=gr.Progress()):
                    return h.run_analysis(img, mods, pre, k, langs, key, mdl, search, nts, img_url, provider_id=prov, progress=progress)

                ai_provider.input(sh.provider_search_default, ai_provider, use_search, api_visibility="private")
                run_btn.click(
                    _run,
                    inputs=[image, modules, preset, top_k, ocr_langs, api_key, model, use_search, notes, url, ai_provider],
                    outputs=outputs,
                    api_name="analyze",
                ).then(h.header_html, outputs=header, api_visibility="private")
                fetch_btn.click(h.fetch_url, inputs=url, outputs=[image, fetch_msg], api_visibility="private")
                url.submit(h.fetch_url, inputs=url, outputs=[image, fetch_msg], api_visibility="private")
                clear_btn.add([image, url, fetch_msg, notes, *outputs])
                clear_btn.click(lambda: h.blank_outputs()[:2], outputs=[verdict, map_html], api_visibility="private")

            # ------------------------------------------------------------------
            with gr.Tab("Coordinate", id="coords"):
                gr.HTML(
                    '<p class="go-intro">Incolla coordinate in qualsiasi formato: decimali, gradi e minuti, UTM, Plus Code, '
                    "Geohash o un link di Google Maps, OpenStreetMap, Bing. Le converto in tutti gli altri e ti do i link per verificarle.</p>"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        coord_in = gr.Textbox(label="Coordinate", value=EXAMPLE_PLACE, placeholder="45.4408, 12.3155")
                        coord_rev = gr.Checkbox(value=s.online_lookups, label="Cerca l'indirizzo su OpenStreetMap")
                        coord_btn = gr.Button("Converti", variant="primary")
                        coord_card = gr.HTML()
                        with gr.Accordion("Cerca un luogo per nome", open=False):
                            place_q = gr.Textbox(placeholder="Indirizzo, monumento, nome di un'attività", show_label=False)
                            place_btn = gr.Button("Cerca luogo")
                            place_df = gr.Dataframe(headers=["Nome", "Lat", "Lon", "Tipo"], interactive=False, wrap=True)
                        with gr.Accordion("Distanza tra due punti", open=False):
                            with gr.Row():
                                dist_a = gr.Textbox(label="Punto A", value=EXAMPLE_PLACE)
                                dist_b = gr.Textbox(label="Punto B", value="41.902167, 12.453937")
                            dist_btn = gr.Button("Misura distanza")
                            dist_out = gr.HTML()
                    with gr.Column(scale=6, min_width=380):
                        coord_map = gr.HTML()
                        coord_links = gr.HTML()
                coord_btn.click(h.coordinate_tool, [coord_in, coord_rev], [coord_card, coord_map, coord_links], api_name="coordinates")
                coord_in.submit(h.coordinate_tool, [coord_in, coord_rev], [coord_card, coord_map, coord_links], api_visibility="private")
                place_btn.click(h.place_search, place_q, [place_df, coord_map], api_name="place_search")
                place_q.submit(h.place_search, place_q, [place_df, coord_map], api_visibility="private")
                dist_btn.click(h.distance_tool, [dist_a, dist_b], dist_out, api_name="distance")

            # ------------------------------------------------------------------
            with gr.Tab("Sole e ombre", id="sun"):
                gr.HTML(
                    '<p class="go-intro">Dove si trovava il sole in un luogo e in un momento, e a che ora un oggetto di altezza nota '
                    "proietta l'ombra che vedi nella foto. Misura altezza e ombra nella foto o sul satellite.</p>"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        sun_coords = gr.Textbox(label="Luogo (coordinate)", value=EXAMPLE_PLACE)
                        with gr.Row():
                            sun_day = gr.Textbox(label="Data (AAAA-MM-GG)", value=date.today().isoformat())
                            sun_time = gr.Textbox(label="Ora locale (HH:MM)", value="12:00")
                        sun_tz = gr.Textbox(
                            label="Fuso orario", value="Europe/Rome", info="Nome IANA, es. Europe/Rome, oppure scostamento, es. +02:00"
                        )
                        sun_btn = gr.Button("Calcola posizione del sole", variant="primary")
                        gr.HTML('<div class="eyebrow" style="margin-top:14px">Dall\'ombra all\'orario</div>')
                        with gr.Row():
                            sh_h = gr.Number(label="Altezza oggetto (m)", value=1.8, minimum=0.01)
                            sh_l = gr.Number(label="Lunghezza ombra (m)", value=3.2, minimum=0.01)
                        sh_az = gr.Number(
                            label="Direzione dell'ombra (gradi da Nord, 0 se non la conosci)", value=0, minimum=0, maximum=360
                        )
                        shadow_btn = gr.Button("Trova gli orari")
                    with gr.Column(scale=6, min_width=380):
                        with gr.Row(equal_height=False):
                            sun_card = gr.HTML()
                            sun_svg = gr.HTML()
                        shadow_out = gr.HTML()
                sun_btn.click(h.sun_tool, [sun_coords, sun_day, sun_time, sun_tz], [sun_card, sun_svg], api_name="sun")
                shadow_btn.click(h.shadow_tool, [sun_coords, sun_day, sun_tz, sh_h, sh_l, sh_az], shadow_out, api_name="shadow")

            # ------------------------------------------------------------------
            with gr.Tab("Confronto", id="compare"):
                gr.HTML(
                    '<p class="go-intro">Due versioni della stessa foto? Gli hash percettivi dicono se sono la stessa immagine '
                    "ricompressa o ritagliata; i metadati indicano quale è più vicina all'originale.</p>"
                )
                with gr.Row():
                    cmp_a = gr.Image(type="filepath", label="Immagine A", height=260)
                    cmp_b = gr.Image(type="filepath", label="Immagine B", height=260)
                cmp_btn = gr.Button("Confronta immagini", variant="primary")
                cmp_out = gr.HTML()
                cmp_btn.click(h.compare_tool, [cmp_a, cmp_b], cmp_out, api_name="compare")

            # ------------------------------------------------------------------
            with gr.Tab("Batch", id="batch"):
                gr.HTML(
                    '<p class="go-intro">Analizza molte foto insieme: ottieni una tabella riepilogativa, una mappa con tutti i punti '
                    "e l'export in CSV e JSON.</p>"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        batch_files = gr.File(label="Foto", file_count="multiple", file_types=["image"])
                        batch_modules = gr.CheckboxGroup(all_modules, value=["Metadati", "Forense", "GeoCLIP"], label="Moduli")
                        gr.HTML("<p class=\"go-hint\">L'arbitro IA su molte foto consuma rapidamente la quota dell'API.</p>")
                        batch_preset = gr.Dropdown(preset_choices, value=s.geoclip_preset, label="Modello GeoCLIP")
                        batch_btn = gr.Button("Avvia analisi batch", variant="primary")
                        batch_out_files = gr.File(label="Export", file_count="multiple", interactive=False)
                    with gr.Column(scale=6, min_width=380):
                        batch_map = gr.HTML(h.map_with_legend([], 400))
                batch_df = gr.Dataframe(headers=h.BATCH_HEADERS, interactive=False, wrap=True)

                def _batch(files, mods, pre, progress=gr.Progress()):
                    return h.batch_tool(files, mods, pre, progress=progress)

                batch_btn.click(
                    _batch, [batch_files, batch_modules, batch_preset], [batch_df, batch_map, batch_out_files], api_name="batch"
                )

            # ------------------------------------------------------------------
            with gr.Tab("Privacy", id="privacy"):
                gr.HTML(
                    '<p class="go-intro">Prima di pubblicare una tua foto: guarda cosa rivela (posizione GPS, numero di serie, autore) '
                    "e scaricane una copia senza metadati.</p>"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        priv_in = gr.Image(type="filepath", label="Foto", height=280)
                        priv_btn = gr.Button("Crea copia pulita", variant="primary")
                        priv_file = gr.File(label="Copia senza metadati", interactive=False)
                    with gr.Column(scale=6, min_width=380):
                        priv_out = gr.HTML(h.privacy_tool(None)[0])
                priv_btn.click(h.privacy_tool, priv_in, [priv_out, priv_file], api_name="strip_metadata")

            # ------------------------------------------------------------------
            settings_tab = build_settings_tab(ai_provider, header, h.header_html)

            # ------------------------------------------------------------------
            with gr.Tab("Guida", id="info"), gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    gr.Markdown(GUIDE)
                with gr.Column(scale=5):
                    status = gr.HTML(h.status_html())
                    gr.Button("Aggiorna stato", size="sm").click(h.status_html, outputs=status, api_visibility="private")

        demo.load(h.header_html, outputs=header, api_visibility="private")
        demo.load(sh.snapshot, outputs=settings_tab["snapshot_outputs"], api_visibility="private")
        # Tools open in a working state: the example place is already computed.
        demo.load(lambda: h.coordinate_tool(EXAMPLE_PLACE, False), outputs=[coord_card, coord_map, coord_links], api_visibility="private")
        demo.load(
            lambda: h.sun_tool(EXAMPLE_PLACE, date.today().isoformat(), "12:00", "Europe/Rome"),
            outputs=[sun_card, sun_svg],
            api_visibility="private",
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    import argparse

    s = get_settings()
    parser = argparse.ArgumentParser(description="Avvia la dashboard web di GeoOSINT")
    parser.add_argument("--host", default=s.server_host, help="indirizzo di ascolto (0.0.0.0 per la rete locale)")
    parser.add_argument("--port", type=int, default=s.server_port)
    parser.add_argument("--share", action="store_true", default=s.share, help="crea un link pubblico temporaneo Gradio")
    parser.add_argument("--auth", help="protezione con credenziali utente:password")
    parser.add_argument("--inbrowser", action="store_true", help="apri automaticamente il browser")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    auth = tuple(args.auth.split(":", 1)) if args.auth and ":" in args.auth else None
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    allowed = [str(s.reports_dir)] + ([str(EXAMPLE_IMAGE.parent)] if EXAMPLE_IMAGE.exists() else [])
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        inbrowser=args.inbrowser,
        theme=build_theme(),
        css=CSS,
        head=HEAD,
        allowed_paths=allowed,
        max_file_size=f"{s.max_image_mb}mb",
    )


if __name__ == "__main__":
    main()
