import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "models/digit_cnn.keras"

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload an image containing one handwritten digit (0-9).")

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Please run `python train.py` first.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=250)

    # Prepare image
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32") / 255.0

    # Shape: (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array, verbose=0)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.2f}%")
