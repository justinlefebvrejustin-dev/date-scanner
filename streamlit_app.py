import streamlit as st
import os
import google.generativeai as genai
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

# ==========================================
# 1. SETUP & STYLE (Originele Mobiele Versie)
# ==========================================
st.set_page_config(page_title="Date Scanner V5", page_icon="📅")
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
    
    with st.spinner('AI is analyzing...'):
        try:
            # Gebruik gemini-1.5-flash (stabiele versie)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            prompt = """Analyze this image very carefully.
            1. Look for any date (expiration date, best before, THT, or EXP).
            2. Identify the product name.
            
            Format your response EXACTLY like this:
            PRODUCT: [product name]
            DATE: [date found or NULL]"""
            
            res = model.generate_content([prompt, image_pil])
            response_text = res.text.strip()
            
            # Robuuste parsing van het antwoord
            for line in response_text.split('\n'):
                clean_line = line.replace('*', '').strip() # Verwijder markdown
                if 'PRODUCT:' in clean_line.upper():
                    product_name_from_ai = clean_line.split(':', 1)[1].strip()
                if 'DATE:' in clean_line.upper():
                    potential_date = clean_line.split(':', 1)[1].strip()
                    # Check of het geen NULL is en of er minstens een cijfer in staat
                    if potential_date.upper() != 'NULL' and any(char.isdigit() for char in potential_date):
                        date_text = potential_date
                        date_found = True
                        
        except Exception as e:
            # Als er een 404 of andere API fout is, loggen we het kort en gaan we door
            st.warning("Gemini is currently unavailable, switching to product tips.")
            print(f"DEBUG: {e}")
    
    # STAP 2: Resultaat tonen
    speak_text = ""
    
    if date_found:
        # DATUM GEVONDEN - Direct tonen en voorlezen
        product_display = product_name_from_ai if product_name_from_ai else "product"
        st.markdown(f'''<div class="success-box">
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Product</div>
            <div style="color:white;font-size:1.6em;font-weight:bold;">{product_display}</div>
            <div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;margin-top:10px;">Expiration Date</div>
            <div style="color:#16a34a;font-size:2.2em;font-weight:900;">{date_text}</div>
            <div style="color:#d1fae5;margin-top:5px;font-weight:bold;">✅ Date detected by AI</div>
        </div>''', unsafe_allow_html=True)
        
        speak_text = f"The date for this {product_display} is {date_text}"
        
    else:
        # GEEN DATUM - Overschakelen naar Teachable Machine voor tips
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
            speak_text = f"I see {product_name}. {tip}"
    
    # AUDIO UITVOER
    if speak_text:
        try:
            tts = gTTS(speak_text, lang='en', tld='com')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        except: pass
