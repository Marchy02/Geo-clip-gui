"""Gradio dashboard."""

from __future__ import annotations

import logging
from datetime import date

import gradio as gr

from .. import __version__
from ..config import get_settings
from ..engines.geoclip_engine import presets
from . import handlers as h
from .maps import empty_map
from .theme import CSS, HEAD, build_theme

GUIDE = """
### Flusso di lavoro consigliato
1. **Analisi** — carica l'immagine e premi *Analizza*. Il verdetto sceglie la fonte più affidabile:
   GPS EXIF ▸ arbitro IA con prove indipendenti ▸ stima GeoCLIP.
2. **Verifica** — non fidarti del verdetto: apri *Street View*, *Mapillary* o il *satellite* dai link
   e confronta edifici, cartelli, pali, vegetazione, orizzonte.
3. **Pivot** — cerca i testi letti dall'OCR, lancia la ricerca inversa per trovare la copia più vecchia
   dell'immagine (spesso con contesto e didascalia originali).
4. **Cronolocalizza** — con la scheda *Sole & ombre* stima data/ora dalla lunghezza e direzione delle ombre.
5. **Documenta** — scarica il report HTML/JSON/Markdown: contiene hash SHA-256 per la catena di custodia.

### Suggerimenti
* Le foto passate da social o chat perdono i metadati: usa la ricerca inversa per trovare un originale.
* GeoCLIP dà una *distribuzione* di probabilità: candidati molto dispersi = scena poco caratteristica.
* Nelle *Note dell'analista* aggiungi ciò che sai (es. "video girato in Est Europa, inverno"): l'IA ne terrà conto.
* Per immagini di Ucraina/Russia scegli il preset regionale di GeoCLIP.
* Rispetta la legge e la privacy delle persone: usa lo strumento solo per indagini legittime.
"""


def build_app() -> gr.Blocks:
    s = get_settings()
    preset_choices = [(desc, key) for key, desc in presets().items()]
    lang_labels = list(h.OCR_LANGS)
    default_langs = [label for label, code in h.OCR_LANGS.items() if code in s.ocr_languages] or ["Inglese"]
    all_modules = list(h.MODULES)

    with gr.Blocks(title=f"GeoOSINT {__version__}") as demo:
        header = gr.HTML(h.header_html())

        with gr.Tabs():
            # ------------------------------------------------------------------
            with gr.Tab("🔍 Analisi immagine", id="analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=340):
                        image = gr.Image(
                            type="filepath",
                            label="Immagine",
                            sources=["upload", "clipboard", "webcam"],
                            height=330,
                        )
                        with gr.Row():
                            url = gr.Textbox(
                                placeholder="…oppure URL di un'immagine", show_label=False, scale=5, container=False, lines=1, max_lines=1
                            )
                            fetch_btn = gr.Button("⬇️ Scarica", scale=1, size="sm")
                        fetch_msg = gr.HTML()
                        with gr.Accordion("⚙️ Opzioni di analisi", open=False):
                            modules = gr.CheckboxGroup(all_modules, value=all_modules, label="Moduli")
                            preset = gr.Dropdown(preset_choices, value=s.geoclip_preset, label="Modello GeoCLIP")
                            top_k = gr.Slider(1, 20, value=s.geoclip_top_k, step=1, label="Candidati GeoCLIP")
                            ocr_langs = gr.Dropdown(lang_labels, value=default_langs, multiselect=True, label="Lingue OCR")
                            with gr.Group():
                                api_key = gr.Textbox(
                                    label="Chiave API Gemini",
                                    type="password",
                                    placeholder="già configurata da .env"
                                    if s.gemini_api_key
                                    else "incolla qui la chiave (non viene salvata)",
                                )
                                with gr.Row():
                                    model = gr.Textbox(value=s.gemini_model, label="Modello Gemini", scale=3)
                                    use_search = gr.Checkbox(value=s.gemini_search, label="Ricerca Google", scale=2)
                            notes = gr.Textbox(
                                label="Note dell'analista (contesto per l'IA)",
                                placeholder="Es. foto pubblicata su Telegram il 3/3, si parla di una città costiera…",
                                lines=2,
                            )
                        with gr.Row():
                            run_btn = gr.Button("🚀 Analizza", variant="primary", scale=3, elem_classes="go-primary")
                            clear_btn = gr.ClearButton(value="Pulisci", scale=1)
                        steps = gr.HTML()

                    with gr.Column(scale=7, min_width=420):
                        verdict = gr.HTML(h.blank_outputs("Carica un'immagine e premi Analizza.")[0])
                        map_html = gr.HTML(empty_map())
                        with gr.Tabs():
                            with gr.Tab("🧭 Geolocalizzazione"):
                                geo_df = gr.Dataframe(
                                    headers=["#", "Lat", "Lon", "Probabilità", "Zona"],
                                    label="Candidati GeoCLIP",
                                    interactive=False,
                                    wrap=True,
                                )
                                geo_html = gr.HTML()
                            with gr.Tab("🤖 Arbitro IA"):
                                ai_html = gr.HTML()
                            with gr.Tab("🗂️ Metadati"):
                                meta_html = gr.HTML()
                                exif_df = gr.Dataframe(
                                    headers=["Tag", "Valore"], label="Tutti i metadati", interactive=False, wrap=True, max_height=420
                                )
                            with gr.Tab("🧪 Forense"):
                                with gr.Row():
                                    ela_img = gr.Image(label="Error Level Analysis", interactive=False, height=300)
                                    noise_img = gr.Image(label="Residuo di rumore", interactive=False, height=300)
                                forensic_html = gr.HTML()
                            with gr.Tab("🔤 Testo & oggetti"):
                                ocr_html = gr.HTML()
                                yolo_img = gr.Image(label="Oggetti rilevati", interactive=False, height=360)
                                yolo_html = gr.HTML()
                            with gr.Tab("🔁 Ricerca inversa"):
                                reverse_html = gr.HTML()
                            with gr.Tab("📄 Report"):
                                gr.Markdown(
                                    "Report generati automaticamente a ogni analisi: **HTML** (dossier), **JSON** (dati), **Markdown** (note)."
                                )
                                report_files = gr.File(label="Download", file_count="multiple", interactive=False)
                                report_json = gr.JSON(label="Dati grezzi", open=False)

                outputs = [
                    verdict, map_html, geo_df, geo_html, ai_html, meta_html, exif_df, ela_img, noise_img,
                    forensic_html, ocr_html, yolo_img, yolo_html, reverse_html, report_files, steps, report_json,
                ]  # fmt: skip

                def _run(img, mods, pre, k, langs, key, mdl, search, nts, img_url, progress=gr.Progress()):
                    return h.run_analysis(img, mods, pre, k, langs, key, mdl, search, nts, img_url, progress=progress)

                run_btn.click(
                    _run,
                    inputs=[image, modules, preset, top_k, ocr_langs, api_key, model, use_search, notes, url],
                    outputs=outputs,
                    api_name="analyze",
                ).then(h.header_html, outputs=header, api_visibility="private")
                fetch_btn.click(h.fetch_url, inputs=url, outputs=[image, fetch_msg], api_visibility="private")
                url.submit(h.fetch_url, inputs=url, outputs=[image, fetch_msg], api_visibility="private")
                clear_btn.add([image, url, fetch_msg, notes, *outputs])

            # ------------------------------------------------------------------
            with gr.Tab("📍 Coordinate & luoghi", id="coords"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        coord_in = gr.Textbox(
                            label="Coordinate in qualsiasi formato",
                            placeholder="45.4408, 12.3155 · 45°26'27\"N 12°18'56\"E · link Google Maps · 8FVC9G8F+6X · 33T 291000 5034000",
                        )
                        coord_rev = gr.Checkbox(value=True, label="Cerca indirizzo (OpenStreetMap)")
                        coord_btn = gr.Button("Converti e localizza", variant="primary")
                        coord_card = gr.HTML()
                        gr.Markdown("#### 🔎 Cerca un luogo")
                        place_q = gr.Textbox(placeholder="Indirizzo, nome di attività, monumento…", show_label=False)
                        place_btn = gr.Button("Cerca")
                        place_df = gr.Dataframe(headers=["Nome", "Lat", "Lon", "Tipo"], interactive=False, wrap=True)
                        gr.Markdown("#### 📏 Distanza tra due punti")
                        with gr.Row():
                            dist_a = gr.Textbox(label="Punto A")
                            dist_b = gr.Textbox(label="Punto B")
                        dist_btn = gr.Button("Calcola")
                        dist_out = gr.HTML()
                    with gr.Column(scale=6, min_width=380):
                        coord_map = gr.HTML(empty_map(380))
                        coord_links = gr.HTML()
                coord_btn.click(h.coordinate_tool, [coord_in, coord_rev], [coord_card, coord_map, coord_links], api_name="coordinates")
                coord_in.submit(h.coordinate_tool, [coord_in, coord_rev], [coord_card, coord_map, coord_links], api_visibility="private")
                place_btn.click(h.place_search, place_q, [place_df, coord_map], api_name="place_search")
                place_q.submit(h.place_search, place_q, [place_df, coord_map], api_visibility="private")
                dist_btn.click(h.distance_tool, [dist_a, dist_b], dist_out, api_name="distance")

            # ------------------------------------------------------------------
            with gr.Tab("☀️ Sole & ombre", id="sun"):
                gr.Markdown(
                    "Cronolocalizzazione: posizione del sole per luogo/data/ora, e ricerca dell'orario a partire da un'ombra "
                    "(altezza di un oggetto noto e lunghezza della sua ombra, misurate nella foto o sul satellite)."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        sun_coords = gr.Textbox(label="Luogo (coordinate)", value="41.8902, 12.4922")
                        with gr.Row():
                            sun_day = gr.Textbox(label="Data (AAAA-MM-GG)", value=date.today().isoformat())
                            sun_time = gr.Textbox(label="Ora locale (HH:MM)", value="12:00")
                        sun_tz = gr.Textbox(label="Fuso orario", value="Europe/Rome", info="Nome IANA (Europe/Rome) oppure offset (+02:00)")
                        sun_btn = gr.Button("Calcola posizione del sole", variant="primary")
                        gr.Markdown("#### 🕰️ Da un'ombra all'orario")
                        with gr.Row():
                            sh_h = gr.Number(label="Altezza oggetto (m)", value=1.8)
                            sh_l = gr.Number(label="Lunghezza ombra (m)", value=3.2)
                            sh_az = gr.Number(label="Direzione ombra (° da N, 0 = ignora)", value=0)
                        shadow_btn = gr.Button("Trova orari compatibili")
                    with gr.Column(scale=6, min_width=380):
                        with gr.Row():
                            sun_card = gr.HTML()
                            sun_svg = gr.HTML()
                        shadow_out = gr.HTML()
                sun_btn.click(h.sun_tool, [sun_coords, sun_day, sun_time, sun_tz], [sun_card, sun_svg], api_name="sun")
                shadow_btn.click(h.shadow_tool, [sun_coords, sun_day, sun_tz, sh_h, sh_l, sh_az], shadow_out, api_name="shadow")

            # ------------------------------------------------------------------
            with gr.Tab("🧩 Confronta", id="compare"):
                gr.Markdown(
                    "Stessa immagine ricompressa/ritagliata? Quale versione è più vicina all'originale? Confronto con hash percettivi e metadati."
                )
                with gr.Row():
                    cmp_a = gr.Image(type="filepath", label="Immagine A", height=280)
                    cmp_b = gr.Image(type="filepath", label="Immagine B", height=280)
                cmp_btn = gr.Button("Confronta", variant="primary")
                cmp_out = gr.HTML()
                cmp_btn.click(h.compare_tool, [cmp_a, cmp_b], cmp_out, api_name="compare")

            # ------------------------------------------------------------------
            with gr.Tab("🗃️ Batch", id="batch"):
                gr.Markdown("Analizza una cartella di immagini in un colpo solo: tabella riepilogativa, mappa comune ed export CSV/JSON.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        batch_files = gr.File(label="Immagini", file_count="multiple", file_types=["image"])
                        batch_modules = gr.CheckboxGroup(
                            all_modules, value=["Metadati", "Forense", "GeoCLIP"], label="Moduli (l'IA su molti file consuma quota API)"
                        )
                        batch_preset = gr.Dropdown(preset_choices, value=s.geoclip_preset, label="Modello GeoCLIP")
                        batch_btn = gr.Button("Avvia batch", variant="primary")
                        batch_out_files = gr.File(label="Export", file_count="multiple", interactive=False)
                    with gr.Column(scale=6, min_width=380):
                        batch_map = gr.HTML(empty_map(400))
                batch_df = gr.Dataframe(headers=h.BATCH_HEADERS, interactive=False, wrap=True)

                def _batch(files, mods, pre, progress=gr.Progress()):
                    return h.batch_tool(files, mods, pre, progress=progress)

                batch_btn.click(
                    _batch, [batch_files, batch_modules, batch_preset], [batch_df, batch_map, batch_out_files], api_name="batch"
                )

            # ------------------------------------------------------------------
            with gr.Tab("🧹 Privacy", id="privacy"):
                gr.Markdown(
                    "Prima di pubblicare una tua foto: scopri cosa rivela (GPS, seriali, autore) e scarica una copia **senza metadati**."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        priv_in = gr.Image(type="filepath", label="Immagine", height=300)
                        priv_btn = gr.Button("Analizza e ripulisci", variant="primary")
                        priv_file = gr.File(label="Copia pulita", interactive=False)
                    with gr.Column(scale=6, min_width=380):
                        priv_out = gr.HTML()
                priv_btn.click(h.privacy_tool, priv_in, [priv_out, priv_file], api_name="strip_metadata")

            # ------------------------------------------------------------------
            with gr.Tab("ℹ️ Guida & stato", id="info"), gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    gr.Markdown(GUIDE)
                with gr.Column(scale=5):
                    status = gr.HTML(h.status_html())
                    gr.Button("Aggiorna stato").click(h.status_html, outputs=status, api_visibility="private")

        demo.load(h.header_html, outputs=header, api_visibility="private")
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
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        inbrowser=args.inbrowser,
        theme=build_theme(),
        css=CSS,
        head=HEAD,
        allowed_paths=[str(s.reports_dir)],
        max_file_size=f"{s.max_image_mb}mb",
    )


if __name__ == "__main__":
    main()
