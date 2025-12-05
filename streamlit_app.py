import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "model.tflite"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

st.title("📈 Candlestick Pattern Market Predictor")
st.write("Upload a candlestick chart image to predict if the next day is bullish or bearish.")

uploaded_file = st.file_uploader("Upload Candlestick Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_index)[0][0]

    label = "📈 Bullish" if prob >= THRESHOLD else "📉 Bearish"

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{prob:.3f}`")
