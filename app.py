# app.py — Interfaccia Grafica
import gradio as gr
import folium
import base64
import io
from analysis import run_full_analysis

# --- RENDERING MAPPA ---
def get_map_base64(lat, lon, popup_text, color):
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles='CartoDB dark_matter')
    folium.Marker(
        [lat, lon], 
        popup=popup_text, 
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)
    
    data = io.BytesIO()
    m.save(data, close_file=False)
    b64 = base64.b64encode(data.getvalue()).decode()
    return f'<iframe src="data:text/html;base64,{b64}" width="100%" height="400px" style="border:none; border-radius:12px;"></iframe>'

# --- GESTIONE PIPELINE ---
def process_ui(img_path):
    if not img_path: 
        return "", "", "", "", "", ""
    
    try:
        res = run_full_analysis(img_path)
        g_data = res.gemini_data

        # --- LOGICA DI OVERRIDE ---
        if g_data.get("override") is True:
            # Gemini ha preso il controllo
            lat = g_data.get("lat")
            lon = g_data.get("lon")
            nome = g_data.get("location_name", "Località Rilevata")
            desc = g_data.get("description", "Dettagli visivi verificati dall'IA.")
            
            map_html = get_map_base64(lat, lon, nome, "red")
            
            verdetto_html = f"""
            <div style='background:#2C1A00; border:2px solid #FFD700; border-radius:8px; padding:15px;'>
                <h3 style='color:#FFD700; margin-top:0;'>🎯 OVERRIDE IA: {nome}</h3>
                <p style='color:#E2E8F0; font-size:0.95rem; margin-bottom:0;'><b>Descrizione Tecnica:</b> {desc}</p>
            </div>
            """
        else:
            # Comando a GeoCLIP
            top_geo = res.geo_predictions[0]
            lat, lon = top_geo.lat, top_geo.lon
            map_html = get_map_base64(lat, lon, "Predizione GeoCLIP", "green")
            
            err_msg = g_data.get("error", "Il luogo non è stato riconosciuto visivamente. Affidarsi ai dati probabilistici.")
            verdetto_html = f"""
            <div style='background:#111827; border:1px solid #374151; border-radius:8px; padding:15px;'>
                <h3 style='color:#00FFB2; margin-top:0;'>🤖 ANALISI STANDARD GEOCLIP</h3>
                <p style='color:#A1A1AA; font-size:0.9rem; margin-bottom:0;'>{err_msg}</p>
            </div>
            """

        # --- FORMATTAZIONE ALTRI DATI ---
        geoclip_raw = "<br>".join([f"#{i+1} {g.lat:.4f}, {g.lon:.4f} ({g.prob*100:.1f}%)" for i, g in enumerate(res.geo_predictions)])
        ocr_raw = ", ".join(res.ocr_texts) if res.ocr_texts else "Nessun testo rilevato."
        yolo_raw = ", ".join(res.yolo_objects) if res.yolo_objects else "Nessun oggetto rilevato."
        exif_raw = "<br>".join([f"<b>{k}:</b> {v}" for k, v in res.exif_info.items()]) if res.exif_info else "Nessun metadato rilevante."

        return (
            map_html, 
            verdetto_html, 
            f"<div class='osint-card'>{geoclip_raw}</div>",
            f"<div class='osint-card'>{ocr_raw}</div>",
            f"<div class='osint-card'>{yolo_raw}</div>",
            f"<div class='osint-card'>{exif_raw}</div>"
        )
    except Exception as e:
        return f"Errore critico: {str(e)}", "", "", "", "", ""

# --- INTERFACCIA GRADIO ---
CUSTOM_CSS = """
body, .gradio-container { background: #080C14 !important; color: #E2E8F0 !important; font-family: 'Space Mono', monospace; }
.osint-card { background:#0D1421; border:1px solid #1E293B; border-radius:8px; padding:10px; margin-bottom:5px; font-size:0.8rem; line-height:1.4; }
"""

with gr.Blocks(title="GeoCLIP OSINT v4.0") as demo:
    gr.HTML("<h1 style='text-align:center; color:#00FFB2;'>◈ GEO·CLIP OSINT v4.0 <span style='color:#FFD700'>[JSON OVERRIDE]</span></h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="Analisi Immagine")
            btn = gr.Button("🚀 ESEGUI ANALISI INTEGRATA", variant="primary")
            gr.HTML("<div style='margin-top:20px;'><b>VERDETTO ARBITRO:</b></div>")
            out_llm = gr.HTML()
            
        with gr.Column(scale=2):
            out_map = gr.HTML()
            with gr.Row():
                with gr.Column(): 
                    gr.HTML("<b>GEOCLIP RAW</b>")
                    out_geo = gr.HTML()
                with gr.Column(): 
                    gr.HTML("<b>OCR & YOLO</b>")
                    out_ocr = gr.HTML()
                    out_yolo = gr.HTML()
                with gr.Column(): 
                    gr.HTML("<b>EXIF METADATA</b>")
                    out_exif = gr.HTML()

    btn.click(
        fn=process_ui, 
        inputs=input_img, 
        outputs=[out_map, out_llm, out_geo, out_ocr, out_yolo, out_exif]
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)