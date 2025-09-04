import streamlit as st
import numpy as np
import pickle
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image

# -------------------------------
# 1. Load Model and Metadata
# -------------------------------
@st.cache_resource
def load_freshguard_model():
    return load_model(r"D:\Projects\FreshGuard\models\freshguard_model.h5")

@st.cache_resource
def load_metadata():
    with open(r"D:\Projects\FreshGuard\models\freshguard_metadata.pkl", "rb") as f:
        return pickle.load(f)

model = load_freshguard_model()
metadata = load_metadata()
int_to_label = metadata["int_to_label"]
IMG_SIZE = (224, 224)

# -------------------------------
# 2. Preprocessing function
# -------------------------------
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict(image):
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0])
    return int_to_label[pred_class], confidence

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="FreshGuard", page_icon="🍎", layout="centered")

st.title("🍎 FreshGuard - Food Freshness Classifier")
st.markdown("Upload an image of a fruit/vegetable to check its **freshness level**.")

uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image (smaller size)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image.resize((300, 300)), caption="Uploaded Image", use_container_width=False)

    # Predict with spinner
    with st.spinner("⏳ Analyzing..."):
        label, confidence = predict(image)

    # Show styled result
    st.success(f"### ✅ Prediction: **{label}** \nConfidence: **{confidence*100:.2f}%**")

# -------------------------------
# Sidebar (About section)
# -------------------------------
st.sidebar.header("ℹ️ About FreshGuard")
st.sidebar.write(
    """
    FreshGuard helps you classify the **freshness level** of fruits & vegetables.  
    - Upload an image 📸  
    - Get an instant prediction ✅  
    - Built with **Deep Learning (Keras + Streamlit)**  
    """
)
