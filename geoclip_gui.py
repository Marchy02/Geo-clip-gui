import gradio as gr
import torch
import folium
from folium.plugins import Fullscreen
from geoclip import GeoCLIP
from PIL import Image, ExifTags
import easyocr

print("Inizializzazione Motori: GeoCLIP & EasyOCR...")
model = GeoCLIP()
# Inizializza l'OCR per le lingue principali europee
reader = easyocr.Reader(['en', 'it', 'de', 'fr', 'es'])

def fix_transformers_output(module, args):
    x = args[0]
    if hasattr(x, 'pooler_output'):
        return (x.pooler_output,)
    if type(x) is tuple:
        return (x[0],)
    return args

model.image_encoder.mlp.register_forward_pre_hook(fix_transformers_output)

def extract_exif(image_path):
    if not image_path:
        return "<div class='exif-box empty'>Nessuna immagine caricata.</div>"
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return "<div class='exif-box empty'>Nessun dato EXIF.<br><span style='font-size:0.65rem; opacity:0.7'>Le immagini scaricate dai social network non contengono metadati.</span></div>"

        exif_dict = {}
        useful_tags = ["Make", "Model", "DateTimeOriginal", "Software", "GPSInfo", "LensModel"]
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag in useful_tags:
                exif_dict[tag] = str(value)[:60]

        if not exif_dict:
            return "<div class='exif-box empty'>EXIF presenti ma senza dati utili (GPS/Device mancanti).</div>"

        html = "<table class='exif-table'>"
        for k, v in exif_dict.items():
            html += f"<tr><td class='exif-key'>{k}</td><td class='exif-val'>{v}</td></tr>"
        html += "</table>"
        return f"<div class='exif-box'>{html}</div>"
    except Exception as e:
        return f"<div class='exif-box error'>Errore EXIF: {str(e)}</div>"

def extract_text(image_path):
    if not image_path:
        return "<div class='exif-box empty'>In attesa dell'immagine...</div>"
    try:
        results = reader.readtext(image_path)
        if not results:
            return "<div class='exif-box empty'>Nessun testo rilevato nell'immagine.</div>"
        
        html = "<ul style='margin:0; padding-left:16px; color:var(--text-primary); font-family:monospace; font-size:0.75rem;'>"
        for (bbox, text, prob) in results:
            if prob > 0.30:  # Filtra i falsi positivi
                html += f"<li style='margin-bottom:4px;'><strong style='color:var(--accent-green)'>{text}</strong> <span style='color:var(--text-muted); font-size:0.65rem;'>(Conf: {prob*100:.0f}%)</span></li>"
        html += "</ul>"
        
        if html == "<ul style='margin:0; padding-left:16px; color:var(--text-primary); font-family:monospace; font-size:0.75rem;'></ul>":
            return "<div class='exif-box empty'>Testo rilevato ma troppo confuso per essere validato.</div>"
            
        return f"<div class='exif-box'>{html}</div>"
    except Exception as e:
        return f"<div class='exif-box error'>Errore OCR: {str(e)}</div>"

def predict(img):
    if img is None:
        m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB dark_matter')
        Fullscreen(position='topright', title='Espandi mappa', title_cancel='Chiudi mappa').add_to(m)
        return m._repr_html_(), []
    try:
        top_pred_gps, top_pred_prob = model.predict(img, top_k=5)
        best_lat, best_lon = float(top_pred_gps[0][0]), float(top_pred_gps[0][1])
        m = folium.Map(location=[best_lat, best_lon], zoom_start=4, tiles='CartoDB dark_matter')
        
        Fullscreen(position='topright', title='Espandi mappa', title_cancel='Chiudi mappa').add_to(m)

        colors = ['#00FFB2', '#0099FF', '#FF6B35', '#FF3CAC', '#FFDD00']
        results_data = []

        for i in range(5):
            lat, lon = float(top_pred_gps[i][0]), float(top_pred_gps[i][1])
            raw_prob = float(top_pred_prob[i])
            prob = min((raw_prob / 0.05) * 100, 99.9)
            
            results_data.append((i + 1, lat, lon, prob))
            color = colors[i]
            
            icon_html = (
                f'<div style="background:{color};width:14px;height:14px;border-radius:50%;'
                f'border:2px solid rgba(255,255,255,0.8);'
                f'box-shadow:0 0 10px {color},0 0 20px {color}60;"></div>'
            )
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(
                    f'<div style="font-family:monospace;color:#0f0;background:#111;padding:6px;border-radius:4px;">'
                    f'<b>PREDICTION #{i+1}</b><br>{lat:.6f}, {lon:.6f}<br><b>{prob:.2f}%</b></div>',
                    max_width=220
                ),
                tooltip=f"#{i+1} — {prob:.2f}%",
                icon=folium.DivIcon(html=icon_html, icon_size=(14, 14), icon_anchor=(7, 7))
            ).add_to(m)

        return m._repr_html_(), results_data

    except Exception as e:
        m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB dark_matter')
        Fullscreen(position='topright', title='Espandi mappa', title_cancel='Chiudi mappa').add_to(m)
        return m._repr_html_(), [("ERROR", str(e), 0, 0)]

def build_results_html(data):
    if not data:
        return "<div class='exif-box empty' style='margin-top:0'>In attesa dell'analisi...</div>"
    if data[0][0] == "ERROR":
        return f"<div class='exif-box error' style='margin-top:0'>ERRORE: {data[0][1]}</div>"

    colors = ['#00FFB2', '#0099FF', '#FF6B35', '#FF3CAC', '#FFDD00']
    max_prob = max(d[3] for d in data)
    rows = ""
    for rank, lat, lon, prob in data:
        color = colors[rank - 1]
        bar_w = f"{(prob / max_prob * 100):.1f}"
        marker = "◈ " if rank == 1 else ""
        name_color = color if rank == 1 else "var(--text-primary)"
        rows += (
            f"<tr>"
            f"<td><span class='rank-badge' style='background:{color}20;color:{color};border:1px solid {color}40'>{rank}</span></td>"
            f"<td style='color:{name_color}'>{marker}<span style='opacity:0.7'>{lat:.4f}°</span> <span style='color:var(--text-muted)'>/</span> <span style='opacity:0.7'>{lon:.4f}°</span></td>"
            f"<td><div class='prob-bar-wrap'>"
            f"<div class='prob-bar'><div class='prob-bar-fill' style='width:{bar_w}%;background:linear-gradient(90deg,{color},{color}80)'></div></div>"
            f"<span class='prob-value' style='color:{color}'>{prob:.2f}%</span>"
            f"</div></td></tr>"
        )
    return (
        "<div class='table-container'>"
        "<table class='results-table'>"
        "<thead><tr><th>#</th><th>Coordinate</th><th>Confidenza</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )

def process_pipeline(img):
    exif_html = extract_exif(img)
    ocr_html = extract_text(img)
    map_html, data = predict(img)
    return map_html, build_results_html(data), exif_html, ocr_html

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root { --bg-primary: #080C14; --bg-secondary: #0D1421; --bg-card: #111827; --accent-green: #00FFB2; --accent-blue: #0EA5E9; --text-primary: #E2E8F0; --text-muted: #64748B; --border: #1E293B; }
* { box-sizing: border-box; }
body, .gradio-container { background: var(--bg-primary) !important; font-family: 'Syne', sans-serif !important; color: var(--text-primary) !important; }
footer { display: none !important; }
.unstyled { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; margin: 0 0 8px 0 !important; min-height: auto !important; }
.header-wrap { text-align:center; padding:30px 20px 20px; position:relative; }
.header-title { font-family:'Syne',sans-serif !important; font-size:2.5rem !important; font-weight:800 !important; letter-spacing:-1px; background:linear-gradient(135deg,#FFF 0%,var(--accent-green) 50%,var(--accent-blue) 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 !important; padding:0 !important; }
.header-sub { font-family:'Space Mono',monospace !important; font-size:0.75rem !important; color:var(--text-muted) !important; letter-spacing:3px; text-transform:uppercase; margin-top:8px !important; }
.gr-group, .gr-box, .block { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:12px !important; }
#image-input { border:2px dashed var(--border) !important; border-radius:12px !important; background:var(--bg-secondary) !important; transition:all 0.3s ease; min-height:180px; margin-bottom: 20px; }
#image-input:hover { border-color:var(--accent-green) !important; box-shadow:0 0 20px rgba(0,255,178,0.3) !important; }
#analyze-btn { background:linear-gradient(135deg,var(--accent-green),var(--accent-blue)) !important; color:#000 !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; font-size:0.85rem !important; letter-spacing:2px; text-transform:uppercase; border:none !important; border-radius:8px !important; padding:12px 24px !important; width:100%; box-shadow:0 4px 20px rgba(0,255,178,0.25); transition:all 0.2s ease; margin-bottom: 20px; }
#analyze-btn:hover { transform:translateY(-2px); box-shadow:0 8px 30px rgba(0,255,178,0.4) !important; }
#map-output { border-radius:10px !important; min-height:400px; overflow:hidden; border:1px solid var(--border); margin-bottom: 20px; }
.table-container { max-height: 250px; overflow-y: auto; background: var(--bg-secondary); border-radius: 8px; border: 1px solid var(--border); }
.results-table { width:100%; border-collapse:collapse; font-family:'Space Mono',monospace; font-size:0.8rem; }
.results-table th { color:var(--text-muted); text-transform:uppercase; letter-spacing:2px; font-size:0.6rem; padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; position: sticky; top: 0; background: var(--bg-card); z-index: 10; }
.results-table td { padding:12px 12px; border-bottom:1px solid rgba(30,41,59,0.5); color:var(--text-primary); }
.results-table tr:hover td { background:rgba(0,255,178,0.04); }
.exif-box { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: 'Space Mono', monospace; overflow-x: auto; max-height: 200px; overflow-y: auto; }
.exif-box.empty { color: var(--text-muted); font-size: 0.75rem; text-align: center; padding: 20px; }
.exif-box.error { color: #FF6B6B; font-size: 0.75rem; }
.exif-table { width: 100%; font-size: 0.7rem; color: var(--text-primary); border-collapse: collapse; }
.exif-key { padding: 4px 8px 4px 0; color: var(--accent-blue); width: 35%; border-bottom: 1px solid rgba(30,41,59,0.5); }
.exif-val { padding: 4px 0; color: var(--text-primary); border-bottom: 1px solid rgba(30,41,59,0.5); word-break: break-all; }
.rank-badge { display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; font-weight:700; font-size:0.65rem; }
.prob-bar-wrap { display:flex; align-items:center; gap:8px; }
.prob-bar { height:4px; border-radius:2px; flex:1; background:var(--border); overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:2px; animation:barGrow 0.8s ease forwards; transform-origin:left; }
@keyframes barGrow { from{transform:scaleX(0)} to{transform:scaleX(1)} }
.prob-value { color:var(--accent-green); font-weight:700; min-width:50px; text-align:right; }
.section-label { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:3px; text-transform:uppercase; color:var(--text-muted); display:flex; align-items:center; gap:8px; }
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }
.status-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--accent-green); box-shadow:0 0 8px var(--accent-green); animation:blink 1.5s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg-primary); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent-green); }
"""

with gr.Blocks(title="GeoCLIP — GeoLocator AI", css=custom_css) as demo:
    gr.HTML("""
    <div class="header-wrap">
        <div class="header-title">◈ GEO·CLIP + OCR</div>
        <div class="header-sub">
            <span class="status-dot"></span>
            &nbsp; Neural Geolocalization Engine &nbsp;·&nbsp; v3.0
        </div>
    </div>
    """, elem_classes="unstyled")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<div class="section-label">01 · Input</div>', elem_classes="unstyled")
            input_img = gr.Image(type="filepath", label="", elem_id="image-input", show_label=False, sources=["upload", "clipboard"])
            
            gr.HTML('<div class="section-label">02 · Analyze</div>', elem_classes="unstyled")
            btn = gr.Button("▶ ESEGUI PIPELINE", variant="primary", elem_id="analyze-btn")
            
            gr.HTML('<div class="section-label">03 · Metadati EXIF</div>', elem_classes="unstyled")
            output_exif = gr.HTML(value="<div class='exif-box empty'>In attesa dell'immagine...</div>", elem_classes="unstyled")
            
            gr.HTML('<br><div class="section-label">04 · Testo Rilevato (OCR)</div>', elem_classes="unstyled")
            output_ocr = gr.HTML(value="<div class='exif-box empty'>In attesa dell'immagine...</div>", elem_classes="unstyled")

        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">05 · Mappa Interattiva</div>', elem_classes="unstyled")
            output_map = gr.HTML(value="<div style='height:400px;background:#0D1421;border-radius:10px;display:flex;align-items:center;justify-content:center;border:1px solid #1E293B'><span style=\"color:#1E293B;font-family:'Space Mono',monospace;font-size:0.8rem;letter-spacing:2px\">MAP OFFLINE — CARICA UN'IMMAGINE</span></div>", elem_classes="unstyled")
            
            gr.HTML('<div class="section-label">06 · Risultati GeoCLIP</div>', elem_classes="unstyled")
            output_text = gr.HTML(value="<div class='exif-box empty'>In attesa dell'analisi...</div>", elem_classes="unstyled")

    btn.click(fn=process_pipeline, inputs=input_img, outputs=[output_map, output_text, output_exif, output_ocr])

if __name__ == "__main__":
    demo.launch()
