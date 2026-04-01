# app.py — Entrypoint. Avvia con: python app.py

import gradio as gr

from analysis import run_full_analysis
from map_render import render_map_html
from ui_helpers import CUSTOM_CSS, to_gradio_outputs

_EMPTY = ("", "", "", "", "", "")


def process_pipeline(img_path: str | None) -> tuple:
    if not img_path:
        return _EMPTY
    try:
        result   = run_full_analysis(img_path)
        map_html = render_map_html(result.geo_predictions)
        return to_gradio_outputs(result, map_html)
    except Exception as exc:
        error = f'<div class="osint-card" style="color:#FF4444;">Errore: {exc}</div>'
        return (error, "", "", "", "", "")


# In Gradio 6 il CSS va passato a launch(), non a Blocks()
with gr.Blocks(title="GeoCLIP OSINT v3.6") as demo:

    gr.HTML("<h1 style='text-align:center;color:#00FFB2;letter-spacing:.15em;'>◈ GEO·CLIP OSINT v3.6</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="filepath", label="Immagine da analizzare")
            run_btn     = gr.Button("🚀 ESEGUI ANALISI INTEGRATA", variant="primary")
            gr.HTML("<br><b>VERDETTO ARBITRO GEMINI</b>")
            out_verdict = gr.HTML()

        with gr.Column(scale=2):
            out_map = gr.HTML()
            out_geo = gr.HTML()
            with gr.Row():
                with gr.Column(): gr.HTML("<b>OCR</b>");  out_ocr  = gr.HTML()
                with gr.Column(): gr.HTML("<b>YOLO</b>"); out_yolo = gr.HTML()
                with gr.Column(): gr.HTML("<b>EXIF</b>"); out_exif = gr.HTML()

    run_btn.click(
        fn=process_pipeline,
        inputs=input_image,
        outputs=[out_map, out_geo, out_exif, out_ocr, out_yolo, out_verdict],
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)  # css qui in Gradio 6