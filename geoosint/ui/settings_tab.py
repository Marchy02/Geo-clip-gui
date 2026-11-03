"""Impostazioni tab: AI providers, general options and the configuration file."""

from __future__ import annotations

import gradio as gr

from ..ai.providers import TEMPLATES
from ..engines.geoclip_engine import presets
from . import settings_handlers as sh
from .handlers import OCR_LANGS

PRIVATE = {"api_visibility": "private"}  # never expose settings changes as public API endpoints


def build_settings_tab(analysis_provider: gr.Dropdown, header: gr.HTML, header_fn) -> dict:
    """Build the tab inside the current ``gr.Tabs`` context; returns components refreshed on page load."""
    s_rows = sh.providers_rows()
    with gr.Tab("Impostazioni", id="settings"):
        gr.HTML(
            '<p class="go-intro">Configura i modelli linguistici che fanno da arbitro IA e le opzioni predefinite '
            "dei motori. Le modifiche valgono dalla prossima analisi e restano salvate.</p>"
        )
        with gr.Tabs():
            # ----------------------------------------------------------
            with gr.Tab("Provider IA", id="settings-ai"):
                gr.HTML(
                    '<p class="go-hint">Le chiamate passano da LiteLLM: oltre ai modelli predefiniti puoi usare qualsiasi '
                    "provider che supporta (Azure, Bedrock, Vertex, DeepSeek, server compatibili OpenAI…). "
                    "Il modello deve accettare immagini.</p>"
                )
                with gr.Row(equal_height=True):
                    active_dd = gr.Dropdown(sh.provider_choices(), value=sh.active_id(), label="Provider predefinito", scale=4)
                    active_btn = gr.Button("Usa come predefinito", scale=1, min_width=160)
                table = gr.Dataframe(
                    value=s_rows,
                    headers=sh.PROVIDER_HEADERS,
                    interactive=False,
                    wrap=True,
                    label="Provider configurati",
                    column_widths=["7%", "19%", "19%", "11%", "22%", "9%", "13%"],
                )

                gr.HTML('<div class="eyebrow" style="margin-top:14px">Aggiungi o modifica</div>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        editor = gr.Dropdown(sh.editor_choices(), value=sh.NEW, label="Provider da modificare")
                        template = gr.Dropdown(list(TEMPLATES), value=None, label="Parti da un modello predefinito")
                        name = gr.Textbox(label="Nome", placeholder="es. Claude lavoro")
                        kind = gr.Dropdown(
                            sh.kind_choices(),
                            value="openai",
                            allow_custom_value=True,
                            label="Tipo di provider (prefisso LiteLLM)",
                            info="Per server compatibili OpenAI scegli openai e indica l'endpoint",
                        )
                        base_url = gr.Textbox(
                            label="Endpoint (facoltativo)",
                            placeholder="vuoto = endpoint ufficiale; es. http://localhost:11434 per Ollama",
                        )
                    with gr.Column(scale=1, min_width=300):
                        with gr.Row(equal_height=True):
                            model = gr.Dropdown([], value="", allow_custom_value=True, label="Modello", scale=4)
                            models_btn = gr.Button("Carica modelli", scale=1, min_width=120, size="sm")
                        api_key = gr.Textbox(label="Chiave API", type="password", placeholder="incolla la chiave")
                        key_hint = gr.HTML(sh.load_provider(sh.NEW)[-1])
                        api_key_env = gr.Textbox(
                            label="…oppure variabile d'ambiente con la chiave",
                            placeholder="es. OPENAI_API_KEY (così la chiave non finisce nel file)",
                        )
                        with gr.Row():
                            web_search = gr.Checkbox(label="Ricerca web", info="Google Search su Gemini, web search su Claude")
                            enabled = gr.Checkbox(value=True, label="Abilitato")
                            make_active = gr.Checkbox(value=False, label="Rendi predefinito")
                with gr.Row():
                    save_btn = gr.Button("Salva provider", variant="primary")
                    test_btn = gr.Button("Prova connessione")
                    delete_btn = gr.Button("Elimina", variant="stop")
                ai_msg = gr.HTML()

            # ----------------------------------------------------------
            with gr.Tab("Generali", id="settings-general"):
                v = dict(zip(sh.GENERAL_ORDER, sh.general_values(), strict=True))
                with gr.Row(equal_height=False):
                    with gr.Column(min_width=300):
                        gr.HTML('<div class="eyebrow">Geolocalizzazione</div>')
                        g_preset = gr.Dropdown(
                            [(desc, key) for key, desc in presets().items()], value=v["geoclip_preset"], label="Modello GeoCLIP predefinito"
                        )
                        g_topk = gr.Slider(1, 20, value=v["geoclip_top_k"], step=1, label="Candidati GeoCLIP")
                        gr.HTML('<div class="eyebrow" style="margin-top:12px">Testo e oggetti</div>')
                        g_langs = gr.Dropdown(list(OCR_LANGS), value=v["ocr_languages"], multiselect=True, label="Lingue OCR predefinite")
                        g_ocr_conf = gr.Slider(0, 1, value=v["ocr_conf"], step=0.05, label="Confidenza minima OCR")
                        g_yolo = gr.Textbox(value=v["yolo_model"], label="Modello YOLO", info="es. yolov8n.pt, yolov8s.pt o un percorso")
                        g_yolo_conf = gr.Slider(0, 1, value=v["yolo_conf"], step=0.05, label="Confidenza minima YOLO")
                    with gr.Column(min_width=300):
                        gr.HTML('<div class="eyebrow">Rete</div>')
                        g_online = gr.Checkbox(value=v["online_lookups"], label="Ricerche online (reverse geocoding OpenStreetMap)")
                        g_http_t = gr.Number(value=v["http_timeout_s"], label="Timeout HTTP (secondi)", minimum=1, maximum=120)
                        g_ua = gr.Textbox(value=v["user_agent"], label="User-Agent delle richieste")
                        g_ai_t = gr.Number(value=v["ai_timeout_s"], label="Timeout arbitro IA (secondi)", minimum=10, maximum=900)
                        gr.HTML('<div class="eyebrow" style="margin-top:12px">File</div>')
                        g_reports = gr.Textbox(value=v["reports_dir"], label="Cartella dei report")
                        g_max_mb = gr.Number(value=v["max_image_mb"], label="Dimensione massima immagine (MB)", minimum=1, maximum=1000)
                general = [
                    g_preset,
                    g_topk,
                    g_langs,
                    g_ocr_conf,
                    g_yolo,
                    g_yolo_conf,
                    g_online,
                    g_http_t,
                    g_ua,
                    g_ai_t,
                    g_reports,
                    g_max_mb,
                ]
                with gr.Row():
                    g_save = gr.Button("Salva impostazioni", variant="primary")
                    g_reset = gr.Button("Ripristina predefiniti")
                g_msg = gr.HTML()

            # ----------------------------------------------------------
            with gr.Tab("File di configurazione", id="settings-file"):
                info = gr.HTML(sh.config_info())
                with gr.Row(equal_height=False):
                    with gr.Column(min_width=300):
                        gr.HTML('<div class="eyebrow">Esporta</div>')
                        exp_secrets = gr.Checkbox(value=False, label="Includi le chiavi API salvate (sconsigliato se condividi il file)")
                        exp_btn = gr.Button("Esporta impostazioni")
                        exp_file = gr.File(label="File esportato", interactive=False)
                    with gr.Column(min_width=300):
                        gr.HTML('<div class="eyebrow">Importa</div>')
                        imp_file = gr.File(label="File JSON di GeoOSINT", file_types=[".json"], type="filepath")
                        imp_btn = gr.Button("Importa e sostituisci")
                file_msg = gr.HTML()

    # ------------------------------------------------------------------
    form = [name, kind, model, base_url, api_key, api_key_env, web_search, enabled]
    form_out = [*form, key_hint]
    shared = [table, editor, active_dd, analysis_provider]

    editor.change(sh.load_provider, [editor, template], form_out, **PRIVATE).then(
        lambda sel: gr.update(visible=sel == sh.NEW), editor, template, **PRIVATE
    )
    template.input(sh.load_provider, [editor, template], form_out, **PRIVATE)
    models_btn.click(sh.fetch_models, [editor, *form], [model, ai_msg], **PRIVATE)
    test_btn.click(sh.test_provider, [editor, *form], ai_msg, **PRIVATE)
    save_btn.click(sh.save_provider, [editor, *form, make_active], [ai_msg, *shared, api_key], **PRIVATE).then(
        header_fn, outputs=header, **PRIVATE
    )
    delete_btn.click(sh.delete_provider, editor, [ai_msg, *shared], **PRIVATE).then(header_fn, outputs=header, **PRIVATE)
    active_btn.click(sh.set_active, active_dd, [ai_msg, *shared], **PRIVATE).then(header_fn, outputs=header, **PRIVATE)

    g_save.click(sh.save_general, general, g_msg, **PRIVATE).then(header_fn, outputs=header, **PRIVATE)
    g_reset.click(sh.reset_general, None, [*general, g_msg], **PRIVATE).then(header_fn, outputs=header, **PRIVATE)

    exp_btn.click(sh.export_config, exp_secrets, exp_file, **PRIVATE)
    imp_btn.click(sh.import_config, imp_file, [file_msg, info, *general, *shared], **PRIVATE).then(header_fn, outputs=header, **PRIVATE)

    return {"snapshot_outputs": [*shared, *general, info]}
