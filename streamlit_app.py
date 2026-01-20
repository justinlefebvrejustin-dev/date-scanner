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

# API KEY
API_KEY = "AIzaSyALqJ7iSB7Ifhy_Ym-b7Hkks5dpMava18I"
genai.configure(api_key=API_KEY)

# ==========================================
# 2. JOUW SPECIFIEKE BIBLIOTHEEK 📚
# ==========================================
TIPS_DB = {
    "Butter": "This is butter. Check the top of the lid.",
    "Soda can": "This is a soda can. The date is usually on the bottom.",
    "Slices of meat": "This is a package with slices of meat. The date is usually on the top of the package.",
    "Milk": "This is milk. The date is usually on the top rim or cap.",
    "Snack": "This is a snack. Look for a white box on the back.",
    "Background": "I don't see a product, please hold it closer to the camera."
}

# ==========================================
# 3. FIX VOOR DEPTHWISECONV2D 🛠️
# ==========================================
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# ==========================================
# 4. CSS VOOR BLAUWE KNOP & DESIGN 🎨
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: white; }
    h1 { color: #3b82f6; text-align: center; font-family: sans-serif; font-weight: 800; }
    
    /* Maak de camera knop blauw */
    div[data-testid="stCameraInput"] button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        padding: 10px !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
    }
    
    /* Styling van de containers */
    div[data-testid="stCameraInput"] { border-radius: 20px; border: 2px solid #333; overflow: hidden; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 20px; border-radius: 15px; margin-top: 20px; }
    
    /* Verberg audio player maar laat hem wel spelen */
    audio { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 5. MODEL LADEN
# ==========================================
@st.cache_resource
def load_model():
    model = None
    labels = []
    try:
        if os.path.exists("keras_model.h5"):
            model = tf.keras.models.load_model("keras_model.h5", 
                                             custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
                                             compile=False)
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return model, labels

model, class_names = load_model()

# ==========================================
# 6. CAMERA & ANALYSE
# ==========================================
# label_visibility="collapsed" voor die strakke HTML-look
img_file = st.camera_input("Scan Product", label_visibility="collapsed")

if img_file is not None:
    # A. VOORBEREIDEN
    image_pil = Image.open(img_file).convert('RGB')
    
    # Gebruik Teachable Machine format (224x224)
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # B. HERKENNEN (Jouw Model)
    product_name = "Background"
    confidence = 0.0
    
    if model:
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        confidence = prediction[0][index]
        
        if confidence > 0.5:
            raw = class_names[index]
            product_name = raw.split(" ", 1)[1] if " " in raw else raw

    # C. DATUM ZOEKEN (Gemini)
    date_text = "NULL"
    if product_name != "Background":
        with st.spinner('Reading date...'):
            try:
                gemini = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Product: {product_name}. Find EXPIRATION DATE. Reply ONLY date in English (e.g. 12 Oct 2025) or 'NULL'."
                res = gemini.generate_content([prompt, image_pil])
                date_text = res.text.strip()
            except: pass

    # D. OUTPUT & AUDIO GENERATIE
    found = "NULL" not in date_text and len(date_text) > 4
    
    # Haal de specifieke tekst uit jouw bibliotheek
    intro_text = TIPS_DB.get(product_name, TIPS_DB["Background"])
    
    if product_name == "Background":
        st.markdown(f"""<div class="error-box"><h3>🔍 No product?</h3><p>{intro_text}</p></div>""", unsafe_allow_html=True)
        speak_text = intro_text
    elif found:
        st.markdown(f"""
        <div class="success-box">
            <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px;">Product</div>
            <div style="color: white; font-size: 1.6em; font-weight: bold;">{product_name}</div>
            <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase; margin-top: 10px;">Expiration Date</div>
            <div style="color: #16a34a; font-size: 2.2em; font-weight: 900;">{date_text}</div>
            <div style="color: #d1fae5; margin-top: 5px; font-weight: bold;">✅ Safe to consume</div>
        </div>
        """, unsafe_allow_html=True)
        speak_text = f"{intro_text} The date is {date_text}."
    else:
        st.markdown(f"""
        <div class="error-box">
            <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase;">Product</div>
            <div style="color: white; font-size: 1.6em; font-weight: bold;">{product_name}</div>
            <div style="color: #dc2626; font-size: 1.5em; font-weight: bold; margin-top: 10px;">Date Not Found</div>
            <div style="color: #fbbf24; margin-top: 5px; font-weight: bold;">💡 {intro_text}</div>
        </div>
        """, unsafe_allow_html=True)
        speak_text = f"No date found. {intro_text}"

    # E. AUDIO (AUTOPLAY)
    try:
        tts = gTTS(speak_text, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            # Autoplay=True zorgt voor direct afspelen
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    except: pass
