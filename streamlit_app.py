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
st.set_page_config(page_title="Date Scanner 0", page_icon="📅")
API_KEY = "AIzaSyALqJ7iSB7Ifhy" + "_Ym-b7Hkks5dpMava18I"
genai.configure(api_key=API_KEY)

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
# 2. MODEL LADEN
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
    
    # STAP 1: GEMINI - Zoek datum
    date_found = False
    date_text = ""
    product_name_from_ai = ""
    
    with st.spinner('Searching for date...'):
        try:
            gemini = genai.GenerativeModel('gemini-1.5-flash')
            prompt = """Look at this product image carefully. Find ANY expiration date, best before date, or use by date visible on the package.
            
The date can be in ANY format: DD/MM/YYYY, DD-MM-YYYY, MM/YYYY, DD.MM.YY, or any other date format.
Also identify what product this is.

Reply EXACTLY in this format:
PRODUCT: [product name]
DATE: [the date you found]

If you cannot find ANY date, reply:
PRODUCT: [product name]
DATE: NULL"""
            
            res = gemini.generate_content([prompt, image_pil])
            response = res.text.strip()
            
            # Parse antwoord
            lines = response.split('\n')
            for line in lines:
                if 'PRODUCT:' in line.upper():
                    product_name_from_ai = line.split(':', 1)[1].strip()
                if 'DATE:' in line.upper():
                    date_text = line.split(':', 1)[1].strip()
            
            # Check of datum echt gevonden is
            if date_text and 'NULL' not in date_text.upper() and len(date_text) > 2:
                date_found = True
                
        except Exception as e:
            pass
    
    # STAP 2: Resultaat tonen
    if date_found:
        # DATUM GEVONDEN - Direct voorlezen
        product_display = product_name_from_ai if product_name_from_ai else "product"
        st.markdown(f'''<div class="success-box">
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Product</div>
            <div style="color:white;font-size:1.6em;font-weight:bold;">{product_display}</div>
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;margin-top:10px;">Expiration Date</div>
            <div style="color:#16a34a;font-size:2.2em;font-weight:900;">{date_text}</div>
            <div style="color:#d1fae5;margin-top:5px;font-weight:bold;">✅ Safe to consume</div>
        </div>''', unsafe_allow_html=True)
        
        speak_text = f"This is {product_display} and the date is {date_text}"
        
    else:
        # GEEN DATUM - Teachable Machine voor tips
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
                confidence = prediction[0][index]
                
                if confidence > 0.5:
                    raw = class_names[index]
                    product_name = raw.split(" ", 1)[1] if " " in raw else raw
            except: pass
        
        tip = TIPS_DB.get(product_name, TIPS_DB["Background"])
        
        if product_name == "Background":
            st.markdown(f'<div class="error-box"><h3>🔍 No product detected</h3><p>{tip}</p></div>', unsafe_allow_html=True)
            speak_text = tip
        else:
            st.markdown(f'''<div class="error-box">
                <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Product</div>
                <div style="color:white;font-size:1.6em;font-weight:bold;">{product_name}</div>
                <div style="color:#dc2626;font-size:1.3em;font-weight:bold;margin-top:10px;">⚠️ No Date Found</div>
                <p style="color:#fbbf24;margin-top:15px;font-size:1.1em;">💡 {tip}</p>
            </div>''', unsafe_allow_html=True)
            speak_text = f"I see this is {product_name}, so {tip}"
    
    # AUDIO (1x afspelen)
    try:
        tts = gTTS(speak_text, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    except: pass
