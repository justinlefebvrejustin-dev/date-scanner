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

# ==========================================
# 2. DE "CLEAN" FIX VOOR DE TENSOR ERROR 🛠️
# ==========================================
@tf.keras.utils.register_keras_serializable()
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# ==========================================
# 3. CSS DESIGN 🎨
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: white; }
    h1 { color: #3b82f6; text-align: center; font-family: sans-serif; font-weight: 800; }
    div[data-testid="stCameraInput"] button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    div[data-testid="stCameraInput"] { border-radius: 20px; border: 2px solid #333; overflow: hidden; }
    .success-box { background: #1f2937; border-left: 8px solid #16a34a; padding: 20px; border-radius: 15px; margin-top: 20px; }
    .error-box { background: #1f2937; border-left: 8px solid #dc2626; padding: 20px; border-radius: 15px; margin-top: 20px; }
    audio { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📅 Date Scanner")

# ==========================================
# 4. MODEL LADEN (De Wrapper Methode)
# ==========================================
@st.cache_resource
def load_model():
    model = None
    labels = []
    try:
        # Wis oude sessies
        tf.keras.backend.clear_session()
        
        if os.path.exists("keras_model.h5"):
            # We laden het model in
            raw_model = tf.keras.models.load_model(
                "keras_model.h5", 
                custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
                compile=False
            )
            
            # DIT IS DE FIX: We maken een nieuwe functie die ALTIJD 1 input geeft
            # en alle extra 'mask' of 'training' troep negeert.
            @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
            def predict_fn(img):
                return raw_model(img, training=False)
            
            model = predict_fn
            
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
            
    return model, labels

model_fn, class_names = load_model()

# ==========================================
# 5. CAMERA & ANALYSE
# ==========================================
img_file = st.camera_input("Scan Product", label_visibility="collapsed")

if img_file is not None:
    image_pil = Image.open(img_file).convert('RGB')
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    
    # Normaliseren
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0) 
    # Converteren naar Tensor voor de wrapper functie
    input_tensor = tf.convert_to_tensor(data)

    product_name = "Background"
    confidence = 0.0
    
    if model_fn:
        try:
            # We roepen de wrapper functie aan ipv het model direct
            prediction = model_fn(input_tensor)
            # Converteren terug naar numpy voor verwerking
            pred_array = prediction.numpy()
            index = np.argmax(pred_array)
            confidence = pred_array[0][index]
            
            if confidence > 0.6:
                raw = class_names[index]
                product_name = raw.split(" ", 1)[1] if " " in raw else raw
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # Gemini Datum Zoeken
    date_text = "NULL"
    if product_name != "Background":
        with st.spinner(f'Identifying {product_name}...'):
            try:
                gemini = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"This image contains {product_name}. Find the EXPIRATION DATE. Reply ONLY the date in English or 'NULL' if not found."
                res = gemini.generate_content([prompt, image_pil])
                date_text = res.text.strip()
            except: pass

    # OUTPUT & AUDIO
    found = "NULL" not in date_text and len(date_text) > 4
    intro_text = TIPS_DB.get(product_name, TIPS_DB["Background"])
    
    if product_name == "Background":
        st.markdown(f"""<div class="error-box"><h3>🔍 No product?</h3><p>{intro_text}</p></div>""", unsafe_allow_html=True)
        speak_text = intro_text
    elif found:
        st.markdown(f"""
        <div class="success-box">
            <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase;">Product Identified</div>
            <div style="color: white; font-size: 1.6em; font-weight: bold; margin-bottom: 5px;">{product_name}</div>
            <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase;">Expiration Date</div>
            <div style="color: #16a34a; font-size: 2.2em; font-weight: 900;">{date_text}</div>
        </div>
        """, unsafe_allow_html=True)
        speak_text = f"{intro_text} The date is {date_text}."
    else:
        st.markdown(f"""
        <div class="error-box">
            <div style="color: #9ca3af; font-size: 0.8em; text-transform: uppercase;">Product Identified</div>
            <div style="color: white; font-size: 1.6em; font-weight: bold; margin-bottom: 5px;">{product_name}</div>
            <div style="color: #dc2626; font-size: 1.5em; font-weight: bold;">Date Not Found</div>
            <p style="color: #fbbf24; margin-top: 10px;">💡 {intro_text}</p>
        </div>
        """, unsafe_allow_html=True)
        speak_text = f"I see the {product_name}, but no date. {intro_text}"

    try:
        tts = gTTS(speak_text, lang='en', tld='com')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    except: pass
