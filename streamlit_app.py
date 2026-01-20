import streamlit as st
import os
import google.generativeai as genai
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile
import re

st.set_page_config(page_title="Date Scanner 1", page_icon="📅")

API_KEY = "AIzaSyBdkCUwIwyYV9Jcu5_ucm3In9A9Z_vx5b4"
genai.configure(api_key=API_KEY)

TIPS_DB = {
    "Butter": "Check the top of the lid.",
    "Soda can": "The date is usually on the bottom.",
    "Slices of meat": "The date is usually on the top of the package.",
    "Milk": "The date is usually on the top rim or cap.",
    "Snack": "Look for a white box on the back.",
    "Background": "Please hold the product closer to the camera."
}

st.title("📅 Date Scanner")

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
    except:
        pass
    return interpreter, labels

interpreter, class_names = load_tflite_model()

img_file = st.camera_input("Scan", label_visibility="collapsed")

def extract_date(text):
    patterns = [
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b",
        r"\b\d{1,2}[./-]\d{4}\b",
        r"\b\d{2}[./-]\d{2}\b"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            return match.group()
    return None

if img_file:
    image_pil = Image.open(img_file).convert("RGB")

    date_found = False
    date_text = ""

    with st.spinner("Reading date..."):
        try:
            gemini = genai.GenerativeModel("gemini-1.5-flash")
            prompt = "Read all visible text on this product. Focus on expiration or best before date."

            res = gemini.generate_content([prompt, image_pil])
            full_text = res.text

            extracted = extract_date(full_text)
            if extracted:
                date_text = extracted
                date_found = True
        except:
            pass

    if date_found:
        st.markdown(f"""
        <div style="background:#1f2937;border-left:8px solid #16a34a;padding:20px;border-radius:15px;">
        <div style="color:#9ca3af;font-size:0.8em;">Expiration Date</div>
        <div style="color:#16a34a;font-size:2.4em;font-weight:900;">{date_text}</div>
        </div>
        """, unsafe_allow_html=True)

        speak_text = f"The date is {date_text}"

    else:
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
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]["index"])
                index = np.argmax(prediction)
                confidence = prediction[0][index]
                if confidence > 0.5:
                    raw = class_names[index]
                    product_name = raw.split(" ", 1)[1] if " " in raw else raw
            except:
                pass

        tip = TIPS_DB.get(product_name, TIPS_DB["Background"])
        st.markdown(f"<div style='color:red'>{tip}</div>", unsafe_allow_html=True)
        speak_text = tip

    try:
        tts = gTTS(speak_text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, autoplay=True)
    except:
        pass
