import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf

# -------------------------------
# 0. GPU setup (optional)
# -------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------------
# 1. Load Model and Metadata
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_freshguard_model():
    model_path = r"D:\Projects\FreshGuard\models\freshguard_model.keras"  # .keras file
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_metadata():
    metadata_path = r"D:\Projects\FreshGuard\models\freshguard_metadata.pkl"
    try:
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Metadata file not found.")
        return None

model = load_freshguard_model()
metadata = load_metadata()

if model is None or metadata is None:
    st.stop()

int_to_label = metadata["int_to_label"]
IMG_SIZE = (224, 224)

# -------------------------------
# 2. Preprocessing function
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict(image: Image.Image):
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
st.markdown("Upload an image of a fruit or vegetable to check its **freshness level**.")

uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image.resize((300, 300)), caption="Uploaded Image", use_container_width=False)
    
    with st.spinner("⏳ Analyzing..."):
        label, confidence = predict(image)

    st.success(f"### ✅ Prediction: **{label}**\nConfidence: **{confidence*100:.2f}%**")

# -------------------------------
# Sidebar (About section)
# -------------------------------
st.sidebar.header("ℹ️ About FreshGuard")
st.sidebar.write(
    """
    FreshGuard helps you classify the **freshness level** of fruits & vegetables. 
    - Upload an image 📸 
    - Get an instant prediction ✅ 
    - Built with **Deep Learning (Keras + Streamlit)** """
)
