import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "best_candle_cnn.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("📈 Candlestick Market Direction Predictor")
st.write("Upload a candlestick image and predict whether the next day is **Bullish** or **Bearish**")

uploaded_file = st.file_uploader("Upload 👇 (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prob = model.predict(x)[0][0]
    label = "Bullish" if prob >= THRESHOLD else "Bearish"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: `{prob:.3f}` (Threshold = {THRESHOLD})")
