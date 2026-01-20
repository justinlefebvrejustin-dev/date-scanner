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
import base64

# ==========================================
# 1. SETUP & STYLE
# ==========================================
st.set_page_config(page_title="Date Scanner V7.2 Pro", page_icon="📅")

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
    .success-box { background: #1f2937; border-left: 10px solid #16a34a; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .error-box { background: #1f2937; border-left: 10px solid #dc2626; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .play-hint { color: #3b82f6; font-size: 0.8em; margin-top: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 2. LOGICA & PREPROCESSING
# ==========================================
def preprocess_for_ocr(image_cv):
    """Verhoogt contrast en verwijdert ruis voor glanzende verpakkingen."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Verwijder schittering door contrast-normalisatie
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def get_human_date(text):
    months = ["", "January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    full_date = re.search(r'(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})', text)
    if full_date:
        d, m, y = map(int, full_date.groups())
        year = y if y > 100 else 2000 + y
        if 1 <= d <= 31 and 1 <= m <= 12:
            suffix = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
            return f"the {d}{suffix} of {months[m]}, {year}", f"{d}/{m}/{year}"

    short_date = re.search(r'(\d{1,2})[./-](\d{4})', text)
    if short_date:
        m, y = map(int, short_date.groups())
        if 1 <= m <= 12:
            return f"{months[m]}, {y}", f"{m}/{y}"
    return None, None

def autoplay_audio(file_path):
    """Hack om audio op mobiel te forceren via Base64."""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=False)
    interpreter = None
    labels = []
    if os.path.exists("model_unquant.tflite"):
        interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
        interpreter.allocate_tensors()
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
    return reader, interpreter, labels

reader, interpreter, class_names = load_models()

# ==========================================
# 3. CAMERA & ANALYSE
# ==========================================
img_file = st.camera_input("Scan", label_visibility="collapsed")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1)
    
    # Verbeter de afbeelding voor glans/vage tekst
    processed_img = preprocess_for_ocr(image_cv)
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    date_found = False
    display_date = ""
    spoken_date = ""

    with st.spinner('Scanning...'):
        try:
            # Scan zowel de originele als de bewerkte foto voor maximale kans
            results = reader.readtext(processed_img) + reader.readtext(image_cv)
            for (bbox, text, prob) in results:
                # Filter op onmogelijke getallen
                if any(c.isdigit() for c in text):
                    human_text, clean_date = get_human_date(text)
                    if human_text:
                        spoken_date = human_text
                        display_date = clean_date
                        date_found = True
                        break 
        except: pass

    if date_found:
        st.markdown(f'''<div class="success-box">
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Result</div>
            <div style="color:white;font-size:1.6em;font-weight:bold;">Date Detected</div>
            <div style="color:#16a34a;font-size:2.5em;font-weight:900;">{display_date}</div>
        </div>''', unsafe_allow_html=True)
        speak_text = f"The expiration date is {spoken_date}"
    else:
        # Teachable Machine Fallback
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
            <p style="color:#fbbf24;margin-top:15px;font-size:1.1em;">💡 Tip: {tip}</p>
        </div>''', unsafe_allow_html=True)
        speak_text = f"I see {product_name}, but no date. {tip}"

    # AUDIO OUTPUT (Special Mobile Fix)
    try:
        tts = gTTS(speak_text, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            autoplay_audio(fp.name)
    except:
        st.write("Audio playback failed. Please check volume.")
