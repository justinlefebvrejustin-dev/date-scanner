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
st.set_page_config(page_title="Date Scanner Pro", page_icon="📅", layout="wide")
API_KEY = "AIzaSyBdkCUwIwyY" + "V9Jcu5_ucm3In9A9Z_vx5b4"
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
    div[data-testid="stCameraInput"] { border-radius: 20px; border: 2px solid #333; overflow: hidden; max-width: 1920px; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 25px; border-radius: 15px; margin-top: 20px; width: 100%; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 25px; border-radius: 15px; margin-top: 20px; width: 100%; }
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
    # Open afbeelding en zorg dat deze het gewenste breedbeeld gevoel behoudt
    image_pil = Image.open(img_file).convert('RGB')
    
    # STAP 1: GEMINI - Zoek datum
    date_found = False
    date_text = ""
    product_name_from_ai = ""
    
    with st.spinner('Scanning for text and dates...'):
        try:
            gemini = genai.GenerativeModel('gemini-1.5-flash')
            # Prompt is aangescherpt voor betere OCR resultaten
            prompt = """Analyze this image. 
            1. Find any expiration date, best before date (THT), or production date. 
            2. Identify the product name.
            
            Return ONLY the following format:
            PRODUCT: [Name]
            DATE: [Found Date or NULL]
            
            If multiple dates are present, pick the one that is likely the expiration date."""
            
            res = gemini.generate_content([prompt, image_pil])
            response_text = res.text.strip()
            
            # Betere parsing van het antwoord
            for line in response_text.split('\n'):
                if line.upper().startswith('PRODUCT:'):
                    product_name_from_ai = line.split(':', 1)[1].strip()
                if line.upper().startswith('DATE:'):
                    date_text = line.split(':', 1)[1].strip()
            
            # Check of de datum bruikbaar is
            if date_text and 'NULL' not in date_text.upper() and len(date_text) >= 5:
                date_found = True
                
        except Exception as e:
            st.error(f"Gemini Error: {e}")
    
    # STAP 2: Resultaat tonen en Spraak genereren
    speak_text = ""
    
    if date_found:
        # DATUM GEVONDEN
        product_display = product_name_from_ai if product_name_from_ai else "Product"
        st.markdown(f'''<div class="success-box">
            <div style="color:#9ca3af;font-size:0.9em;text-transform:uppercase;">Detected Product</div>
            <div style="color:white;font-size:1.8em;font-weight:bold;">{product_display}</div>
            <hr style="border:0.5px solid #374151;margin:15px 0;">
            <div style="color:#9ca3af;font-size:0.9em;text-transform:uppercase;">Expiration Date</div>
            <div style="color:#16a34a;font-size:2.5em;font-weight:900;">{date_text}</div>
            <div style="color:#d1fae5;margin-top:10px;font-size:1.2em;font-weight:bold;">✅ Date found successfully</div>
        </div>''', unsafe_allow_html=True)
        
        speak_text = f"The date for this {product_display} is {date_text}"
        
    else:
        # GEEN DATUM - Overschakelen naar Teachable Machine
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
            st.markdown(f'<div class="error-box"><h3>🔍 No product or date detected</h3><p>{tip}</p></div>', unsafe_allow_html=True)
            speak_text = "I couldn't find a date. " + tip
        else:
            st.markdown(f'''<div class="error-box">
                <div style="color:#9ca3af;font-size:0.9em;text-transform:uppercase;">Detected Product</div>
                <div style="color:white;font-size:1.8em;font-weight:bold;">{product_name}</div>
                <div style="color:#dc2626;font-size:1.4em;font-weight:bold;margin-top:15px;">⚠️ No Date Found</div>
                <p style="color:#fbbf24;margin-top:15px;font-size:1.2em;border-top:1px solid #374151;padding-top:10px;">💡 <b>Tip:</b> {tip}</p>
            </div>''', unsafe_allow_html=True)
            speak_text = f"I see the {product_name}, but no date. {tip}"
    
    # AUDIO UITVOER
    if speak_text:
        try:
            tts = gTTS(speak_text, lang='en', tld='com')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.error(f"TTS Error: {e}")
