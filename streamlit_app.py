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
st.set_page_config(page_title="Date Scanner 3", page_icon="📅")
API_KEY = "AIzaSyALqJ7iSB7Ifhy" + "_Ym-b7Hkks5dpMava18I"
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
    div[data-testid="stCameraInput"] { border-radius: 20px; border: 2px solid #333; overflow: hidden; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 20px; border-radius: 15px; margin-top: 20px; }
    audio { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 2. TFLITE MODEL LADEN
# ==========================================
@st.cache_resource
def load_tflite_model():
    interpreter = None
    labels = []
    model_path = "model_unquantized.tflite"
    try:
        if os.path.exists(model_path):
            # TFLite Interpreter is veel lichter en stabieler
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            st.success("✅ Model loaded successfully!")
        else:
            st.error(f"❌ Model file not found: {model_path}")
            
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
            st.info(f"📋 Loaded {len(labels)} labels: {labels}")
        else:
            st.error("❌ labels.txt not found")
    except Exception as e:
        st.error(f"Error: {e}")
    return interpreter, labels

interpreter, class_names = load_tflite_model()

# ==========================================
# 3. ANALYSE
# ==========================================
img_file = st.camera_input("Scan", label_visibility="collapsed")

if img_file:
    image_pil = Image.open(img_file).convert('RGB')
    
    # Pre-processing (224x224)
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized).astype(np.float32)
    normalized = (image_array / 127.5) - 1
    input_data = np.expand_dims(normalized, axis=0)

    product_name = "Background"
    confidence = 0.0
    
    if interpreter:
        try:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # DEBUG: Toon input/output shapes
            st.write(f"🔍 DEBUG - Input shape expected: {input_details[0]['shape']}")
            st.write(f"🔍 DEBUG - Input shape provided: {input_data.shape}")
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
            # DEBUG: Toon alle predictions
            st.write(f"🔍 DEBUG - All predictions: {prediction[0]}")
            st.write(f"🔍 DEBUG - Best match index: {index}, confidence: {confidence:.2%}")
            
            if confidence > 0.5:  # Lager confidence threshold voor testen
                raw = class_names[index]
                product_name = raw.split(" ", 1)[1] if " " in raw else raw
                st.write(f"✅ Detected: {product_name} ({confidence:.2%})")
            else:
                st.write(f"⚠️ Low confidence: {class_names[index]} ({confidence:.2%})")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Gemini Datum Zoeken
    date_text = "NULL"
    if product_name != "Background":
        with st.spinner(f'Analyzing {product_name}...'):
            try:
                gemini = genai.GenerativeModel('gemini-1.5-flash')
                res = gemini.generate_content([f"Find the expiration date on this {product_name}. Reply ONLY with the date or NULL.", image_pil])
                date_text = res.text.strip()
            except: pass

    # Resultaat & Audio
    found = "NULL" not in date_text and len(date_text) > 4
    intro = TIPS_DB.get(product_name, TIPS_DB["Background"])
    
    if product_name == "Background":
        st.markdown(f'<div class="error-box"><h3>🔍 No product?</h3><p>{intro}</p></div>', unsafe_allow_html=True)
        speak = intro
    elif found:
        st.markdown(f'<div class="success-box"><div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Product</div><div style="color:white;font-size:1.6em;font-weight:bold;">{product_name}</div><div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;margin-top:10px;">Expiration Date</div><div style="color:#16a34a;font-size:2.2em;font-weight:900;">{date_text}</div></div>', unsafe_allow_html=True)
        speak = f"{intro} The date is {date_text}"
    else:
        st.markdown(f'<div class="error-box"><div style="color:#9ca3af;font-size:0.8em;text-transform:uppercase;">Product</div><div style="color:white;font-size:1.6em;font-weight:bold;">{product_name}</div><div style="color:#dc2626;font-size:1.5em;font-weight:bold;margin-top:10px;">Date Not Found</div><p style="color:#fbbf24;margin-top:10px;">💡 {intro}</p></div>', unsafe_allow_html=True)
        speak = f"I see the {product_name}, but no date. {intro}"

    try:
        tts = gTTS(speak, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    except: pass
