import streamlit as st
import os
import google.generativeai as genai
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

# ==========================================
# 1. SETUP & STYLE
# ==========================================
st.set_page_config(page_title="Date Scanner 5", page_icon="📅")
# Gebruik de API key uit je Colab voorbeeld
API_KEY = "AIzaSyALqJ7iSB7Ifhy_Ym-b7Hkks5dpMava18I"
genai.configure(api_key=API_KEY)

TIPS_DB = {
    "Butter": "This is butter. Check the top of the lid.",
    "Soda can": "This is a soda can. The date is usually on the bottom.",
    "Slices of meat": "This is a package with slices of meat. The date is usually on the top of the package.",
    "Milk": "This is milk. The date is usually on the top rim or cap.",
    "Snack": "This is a snack. Look for a white box on the back.",
    "Background": "I don't see a product, please hold it closer to the camera."
}

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: white; }
    h1 { color: #3b82f6; text-align: center; font-family: sans-serif; font-weight: 800; }
    div[data-testid="stCameraInput"] button { background-color: #3b82f6 !important; color: white !important; font-weight: bold; border-radius: 10px; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 20px; border-radius: 15px; margin-top: 20px; }
    audio { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 2. MODEL LADEN (TFLITE)
# ==========================================
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

interpreter, class_names = load_tflite_model()

# ==========================================
# 3. CAMERA & ANALYSE
# ==========================================
img_file = st.camera_input("Scan", label_visibility="collapsed")

if img_file:
    image_pil = Image.open(img_file).convert('RGB')
    speak_text = ""
    
    with st.spinner('🔍 Searching for date...'):
        try:
            # We gebruiken de krachtige prompt uit de Colab versie
            gemini = genai.GenerativeModel('gemini-1.5-flash')
            prompt = """
            Find the expiration date (EXP/BBE) on this product.
            
            INSTRUCTIONS:
            1. Date found? -> Write ONLY the date in English words. 
               (Example: "October twenty-fifth two thousand twenty-four")
            2. No date found? -> Answer EXACTLY: "No date found".
            """
            
            res = gemini.generate_content([prompt, image_pil])
            tekst_resultaat = res.text.strip()
            
            if "No date found" not in tekst_resultaat:
                # ✅ DATUM GEVONDEN
                st.markdown(f'''<div class="success-box">
                    <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Expiration Date</div>
                    <div style="color:#16a34a;font-size:2.2em;font-weight:900;">{tekst_resultaat}</div>
                    <div style="color:#d1fae5;margin-top:5px;font-weight:bold;">✅ Date successfully detected</div>
                </div>''', unsafe_allow_html=True)
                
                speak_text = f"The date is {tekst_resultaat}"
                
            else:
                # ❌ GEEN DATUM - Overschakelen naar Teachable Machine
                size = (224, 224)
                image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image_resized).astype(np.float32)
                normalized = (image_array / 127.5) - 1
                input_data = np.expand_dims(normalized, axis=0)
                
                product_name = "Background"
                if interpreter:
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    index = np.argmax(prediction)
                    
                    if prediction[0][index] > 0.5:
                        raw = class_names[index]
                        product_name = raw.split(" ", 1)[1] if " " in raw else raw

                tip = TIPS_DB.get(product_name, TIPS_DB["Background"])
                
                st.markdown(f'''<div class="error-box">
                    <div style="color:#dc2626;font-size:1.3em;font-weight:bold;">⚠️ No Date Found</div>
                    <p style="color:white;font-size:1.1em;margin-top:10px;">I recognize this as: <b>{product_name}</b></p>
                    <p style="color:#fbbf24;margin-top:5px;">💡 {tip}</p>
                </div>''', unsafe_allow_html=True)
                
                speak_text = f"No date found. {tip}"

        except Exception as e:
            st.error(f"Error: {e}")

    # AUDIO AFSPELEN
    if speak_text:
        try:
            tts = gTTS(speak_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        except: pass
