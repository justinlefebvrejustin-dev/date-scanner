import streamlit as st
import os
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageOps
import tensorflow as tf
from gtts import gTTS
import tempfile
import re

# ==========================================
# 1. SETUP & STYLE (Originele Mobiele Versie)
# ==========================================
st.set_page_config(page_title="Date Scanner V5 - Local OCR", page_icon="📅")

TIPS_DB = {
    "Butter": "Check the top of the lid.",
    "Soda can": "The date is usually on the bottom.",
    "Slices of meat": "The date is usually on the top of the package.",
    "Milk": "The date is usually on the top rim or cap.",
    "Snack": "Look for a white box on the back.",
    "Background": "Please hold the product closer to the camera."
}

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: white; }
    h1 { color: #3b82f6; text-align: center; font-family: sans-serif; font-weight: 800; }
    div[data-testid="stCameraInput"] button { background-color: #3b82f6 !important; color: white !important; font-weight: bold; border-radius: 10px; }
    div[data-testid="stCameraInput"] { border-radius: 20px; border: 2px solid #333; overflow: hidden; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 20px; border-radius: 15px; margin-top: 20px; }
    audio { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 2. MODELLEN LADEN (Cached)
# ==========================================
@st.cache_resource
def load_ocr_reader():
    # Laadt de lokale OCR (Nederlands en Engels)
    return easyocr.Reader(['nl', 'en'], gpu=False)

@st.cache_resource
def load_tflite_model():
    interpreter = None
    labels = []
    try:
        if os.path.exists("model_unquant.tflite"):
            interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
            interpreter.allocate_tensors()
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
    except: pass
    return interpreter, labels

reader = load_ocr_reader()
interpreter, class_names = load_tflite_model()

# ==========================================
# 3. CAMERA & ANALYSE
# ==========================================
img_file = st.camera_input("Scan", label_visibility="collapsed")

if img_file:
    # Converteren naar OpenCV formaat voor EasyOCR
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    date_found = False
    date_text = ""
    all_detected_text = []

    with st.spinner('Lokaal scannen naar datum...'):
        try:
            # STAP 1: Lokale OCR uitvoeren
            results = reader.readtext(image_cv)
            
            # Alle gevonden tekst verzamelen
            for (bbox, text, prob) in results:
                all_detected_text.append(text)
                
                # Direct zoeken naar datum patronen (bv. 12/10/2025 of 12-2025)
                # Deze Regex is zeer breed voor Europese datumnotaties
                match = re.search(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})|(\d{1,2}[./-]\d{2,4})', text)
                if match:
                    date_text = match.group(0)
                    date_found = True
                    break # We hebben een datum, stop met zoeken
        except Exception as e:
            st.error(f"OCR Fout: {e}")

    # STAP 2: RESULTAAT TONEN
    if date_found:
        st.markdown(f'''<div class="success-box">
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Status</div>
            <div style="color:white;font-size:1.6em;font-weight:bold;">Datum Gevonden</div>
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;margin-top:10px;">Vervaldatum</div>
            <div style="color:#16a34a;font-size:2.2em;font-weight:900;">{date_text}</div>
            <div style="color:#d1fae5;margin-top:5px;font-weight:bold;">✅ Lokaal herkend</div>
        </div>''', unsafe_allow_html=True)
        speak_text = f"The expiration date is {date_text}"
        
    else:
        # GEEN DATUM GEVONDEN -> Gebruik Teachable Machine voor Tips
        size = (224, 224)
        image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized).astype(np.float32)
        normalized = (image_array / 127.5) - 1
        input_data = np.expand_dims(normalized, axis=0)
        
        product_name = "Background"
        if interpreter:
            try:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                index = np.argmax(prediction)
                if prediction[0][index] > 0.5:
                    raw = class_names[index]
                    product_name = raw.split(" ", 1)[1] if " " in raw else raw
            except: pass
        
        tip = TIPS_DB.get(product_name, TIPS_DB["Background"])
        st.markdown(f'''<div class="error-box">
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Product</div>
            <div style="color:white;font-size:1.6em;font-weight:bold;">{product_name}</div>
            <div style="color:#dc2626;font-size:1.3em;font-weight:bold;margin-top:10px;">⚠️ Geen datum herkend</div>
            <p style="color:#fbbf24;margin-top:15px;font-size:1.1em;">💡 Tip: {tip}</p>
        </div>''', unsafe_allow_html=True)
        speak_text = f"I see {product_name}, but no date. {tip}"

    # AUDIO VOORLEZEN
    try:
        tts = gTTS(speak_text, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    except: pass

    # Optioneel: Laat zien wat er gelezen is als er niets gevonden werd (voor debugging)
    if not date_found and all_detected_text:
        with st.expander("Gezien tekst (Debug)"):
            st.write(all_detected_text)
