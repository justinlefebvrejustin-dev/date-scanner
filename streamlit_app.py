import streamlit as st
import os
import google.generativeai as genai
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(page_title="Date Scanner", page_icon="📅")
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
    h1 { color: #3b82f6; text-align: center; }
    div[data-testid="stCameraInput"] button { background-color: #3b82f6 !important; color: white !important; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 20px; border-radius: 15px; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 20px; border-radius: 15px; }
    audio { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 2. MODEL LADEN (COLAB STIJL)
# ==========================================
@st.cache_resource
def load_model():
    model = None
    labels = []
    if os.path.exists("keras_model.h5"):
        # In versie 2.15.0 werkt dit gewoon direct
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
    return model, labels

model, class_names = load_model()

# ==========================================
# 3. ANALYSE
# ==========================================
img_file = st.camera_input("Scan", label_visibility="collapsed")

if img_file:
    image_pil = Image.open(img_file).convert('RGB')
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)

    product_name = "Background"
    if model:
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        if prediction[0][index] > 0.6:
            raw = class_names[index]
            product_name = raw.split(" ", 1)[1] if " " in raw else raw

    date_text = "NULL"
    if product_name != "Background":
        with st.spinner('Reading date...'):
            try:
                gemini = genai.GenerativeModel('gemini-1.5-flash')
                res = gemini.generate_content([f"Date on {product_name}?", image_pil])
                date_text = res.text.strip()
            except: pass

    # UI Output
    intro = TIPS_DB.get(product_name, TIPS_DB["Background"])
    if "NULL" not in date_text and len(date_text) > 4:
        st.markdown(f'<div class="success-box"><h3>{product_name}</h3><h2 style="color:#16a34a">{date_text}</h2></div>', unsafe_allow_html=True)
        speak = f"{intro} The date is {date_text}"
    else:
        st.markdown(f'<div class="error-box"><h3>{product_name}</h3><p>{intro}</p></div>', unsafe_allow_html=True)
        speak = f"{intro}"

    try:
        tts = gTTS(speak, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, autoplay=True)
    except: pass
