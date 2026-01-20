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
st.set_page_config(page_title="Date Scanner", page_icon="📅")

# API KEY (Normaal via secrets, maar voor nu hier hardcoded voor gemak)
API_KEY = "AIzaSyALqJ7iSB7Ifhy_Ym-b7Hkks5dpMava18I"
genai.configure(api_key=API_KEY)

# Custom CSS voor de "App Look"
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: white; }
    h1 { color: #3b82f6; text-align: center; }
    .stCameraInput { border-radius: 20px; border: 2px solid #333; }
    div[data-testid="stImage"] img { border-radius: 15px; }
    .success-box { background: #1f2937; border-left: 6px solid #16a34a; padding: 20px; border-radius: 10px; margin-top: 10px; }
    .error-box { background: #1f2937; border-left: 6px solid #dc2626; padding: 20px; border-radius: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 2. MODEL LADEN (Caching voor snelheid)
# ==========================================
@st.cache_resource
def load_model():
    model = None
    labels = []
    try:
        if os.path.exists("keras_model.h5"):
            model = tf.keras.models.load_model("keras_model.h5", compile=False)
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return model, labels

model, class_names = load_model()

# ==========================================
# 3. CAMERA & LOGICA
# ==========================================
img_file = st.camera_input("Maak een foto", label_visibility="hidden")

if img_file is not None:
    # A. VOORBEREIDEN
    image_pil = Image.from(img_file).convert('RGB')
    
    # Resize voor TM (224x224)
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # B. HERKENNEN (Jouw Model)
    product_name = "Unknown"
    confidence = 0.0
    
    with st.spinner('Analyseren...'):
        if model:
            prediction = model.predict(data, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
            if confidence > 0.5:
                raw = class_names[index]
                product_name = raw.split(" ", 1)[1] if " " in raw else raw

        # C. DATUM (Gemini)
        date_text = "NULL"
        if product_name != "Unknown":
            try:
                gemini = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Product: {product_name}. Find EXPIRATION DATE. Reply ONLY date in English (e.g. 12 Oct 2025) or 'NULL'."
                res = gemini.generate_content([prompt, image_pil])
                date_text = res.text.strip()
            except: pass

        # D. RESULTAAT
        found = "NULL" not in date_text and len(date_text) > 4
        
        if found:
            st.markdown(f"""
            <div class="success-box">
                <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase;">Product</div>
                <div style="color: white; font-size: 1.6em; font-weight: bold;">{product_name}</div>
                <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase; margin-top: 10px;">Expiration Date</div>
                <div style="color: #16a34a; font-size: 2em; font-weight: bold;">{date_text}</div>
                <div style="color: #d1fae5; margin-top: 5px;">✅ Safe to consume</div>
            </div>
            """, unsafe_allow_html=True)
            speak_text = f"This is {product_name}. The date is {date_text}."
        else:
            display_name = product_name if product_name != "Unknown" else "Object"
            st.markdown(f"""
            <div class="error-box">
                <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase;">Product</div>
                <div style="color: white; font-size: 1.6em; font-weight: bold;">{display_name}</div>
                <div style="color: #dc2626; font-size: 1.5em; font-weight: bold; margin-top: 10px;">Date Not Found</div>
                <div style="color: #fbbf24; margin-top: 5px;">💡 Try rotating the product</div>
            </div>
            """, unsafe_allow_html=True)
            speak_text = f"This is {display_name}. No date found."

        # Audio Speler
        try:
            tts = gTTS(speak_text, lang='en', tld='com')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", start_time=0)
        except: pass
