import gdown

MODEL_PATH = "vgg16_brain_tumor.h5"
MODEL_URL = "https://drive.google.com/drive/folders/1tEuPqkmm3148017uJEF-L6hERuQaCp6n?usp=sharing"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO


# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Brain Tumor Classification — Hero", layout="wide", page_icon="🧠")
IMAGE_SIZE = 128
MODEL_PATH = "vgg16_brain_tumor.h5"
TRAIN_IMG_DIR = "processed_dataset/training"  # used for decorative left grid

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def get_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return load_model(path)

try:
    model = get_model(MODEL_PATH)
except Exception as e:
    # allow app to start even if model missing; show message later
    model = None
    model_load_error = str(e)
else:
    model_load_error = None

# ---------------------------
# Helper functions
# ---------------------------
def get_decorative_images(folder, max_images=8, thumb_size=(220,220)):
    imgs = []
    if not os.path.isdir(folder):
        return imgs
    # collect up to max_images from subfolders
    for cls in sorted(os.listdir(folder)):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                try:
                    p = os.path.join(cls_dir, f)
                    im = Image.open(p).convert("L")  # grayscale like MRI
                    im = ImageOps.fit(im, thumb_size)
                    imgs.append(im.convert("RGB"))
                    if len(imgs) >= max_images:
                        return imgs
                except:
                    continue
    return imgs

def predict_image(image):
    if model is None:
        raise RuntimeError("Model not loaded.")
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx] * 100.0)
    return preds, idx, conf

def load_class_names(folder):
    if not os.path.isdir(folder):
        return []
    classes = sorted(os.listdir(folder))
    return classes

# ---------------------------
# Styles (hero dark poster + modern UI)
# ---------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: #0b0b0d;
        color: #e6eef8;
    }
    /* hero split */
    .hero {
        display: flex;
        gap: 24px;
        align-items: stretch;
    }
    .left-hero {
        flex: 1.1;
        background: #000000;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.6);
    }
    .right-hero {
        flex: 1;
        padding: 36px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .poster-title {
        font-family: "Segoe UI", Roboto, Arial;
        color: #f3e8ff;
    }
    .poster-title .big-1 { font-size:86px; font-weight:800; color:#ffffff; margin:0; letter-spacing:-1px; }
    .poster-title .big-2 { font-size:80px; font-weight:900; color:#c084fc; margin: -8px 0 0 0; line-height:0.9; }
    .poster-title .big-3 { font-size:46px; font-weight:900; color:#c084fc; margin: -6px 0 12px 0; line-height:0.9; }
    .poster-title .subtitle-hero { color:#a7b3d9; margin-top:10px; font-size:24px; }

    .upload-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius:12px;
        padding:18px;
        box-shadow: 0 8px 40px rgba(12,12,20,0.6);
        border: 1px solid rgba(255,255,255,0.04);
    }
    .muted { color:#9aa7c7; font-size:13px; }
    .result-box {
        margin-top:16px;
        padding:14px;
        border-radius:10px;
        background: #071027;
        border: 1px solid rgba(120,90,220,0.12);
    }
    .confidence-bar {
        height: 12px;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        overflow: hidden;
    }
    .confidence-fill {
        height:100%;
        background: linear-gradient(90deg, #7c3aed, #c084fc);
        border-radius:8px;
        transition: width 0.8s ease;
    }
    .class-pill {
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background: rgba(255,255,255,0.03);
        color:#dbeafe;
        font-weight:600;
        font-size:13px;
        margin-right:8px;
    }
    /* responsive fallback */
    @media (max-width:900px) {
        .hero { flex-direction: column; }
        .left-hero, .right-hero { padding:18px; }
        .big-2 { font-size:48px; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Hero layout
# ---------------------------
classes = load_class_names(TRAIN_IMG_DIR)
int_to_label = {i: label for i, label in enumerate(classes)}
decor_imgs = get_decorative_images(TRAIN_IMG_DIR, max_images=8)

def image_to_base64(image_path):
    img = Image.open(image_path)

    buffer = BytesIO()
    img.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()


st.markdown("<div class='hero'>", unsafe_allow_html=True)

# --- RIGHT SIDE FIRST (TITLE) ---
right_html = """
<div class='right-hero'>
    <div class='poster-title'>
        <h1 class='big-1'>FINAL PROJECT</h1>
        <p class='big-2'>brain tumor classification</p>
        <p class='subtitle-hero'>
            AI-assisted MRI screening • Research tool • Not a clinical device
        </p>
    </div>
</div>
"""
st.markdown(right_html, unsafe_allow_html=True)

# --- LEFT SIDE SECOND (IMAGE) ---
hero_b64 = image_to_base64("brain tumor.jpeg")
left_html = f"""
<div class='left-hero'>
    <img src="data:image/jpeg;base64,{hero_b64}"
         style="
            width:30%;
            height:auto;
            border-radius:16px;
            object-fit:cover;
            box-shadow:0 4px 20px rgba(0,0,0,0.4);
         ">
</div>
"""
st.markdown(left_html, unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)  # close hero


# ---------------------------
# Main content (upload & prediction) under hero
# ---------------------------
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload MRI Scan", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Supported: JPG / JPEG / PNG • Recommended: square or axial slices</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # status of model
    if model_load_error:
        st.error(f"Model load failed: {model_load_error}")
        st.info("Place the trained model file at: " + MODEL_PATH)
    else:
        st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Quick Test", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Upload an MRI to run the VGG16 model and get prediction probabilities.</div>", unsafe_allow_html=True)

        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, width=380)
            except Exception as e:
                st.error("Cannot open image: " + str(e))
                img = None

            if img:
                if st.button("Analyze MRI"):
                    with st.spinner("Running inference..."):
                        try:
                            preds, idx, conf = predict_image(img)
                            label = int_to_label.get(idx, f"Class {idx}")
                        except Exception as e:
                            st.error("Prediction error: " + str(e))
                            preds = None
                            label = None
                            conf = None

                    if preds is not None:
                        # results box
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center;'><div><strong style='color:#cbd5ff'>Prediction</strong><div style='margin-top:6px'><span class='class-pill'>{label}</span></div></div><div style='text-align:right; color:#cbd5ff'><small>Confidence</small><div style='font-weight:800; font-size:20px'>{conf:.2f}%</div></div></div>", unsafe_allow_html=True)

                        # confidence bar
                        st.markdown(f"<div style='margin-top:12px' class='confidence-bar'><div class='confidence-fill' style='width:{conf}%;'></div></div>", unsafe_allow_html=True)

                        # detailed probs
                        st.markdown("<div style='margin-top:14px'><strong style='color:#cbd5ff'>Probabilities</strong></div>", unsafe_allow_html=True)
                        # show per-class horizontal bars
                        for i, c in enumerate(classes):
                            p = preds[i] * 100
                            color = "#c084fc" if i==idx else "rgba(255,255,255,0.08)"
                            st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center; margin-top:8px'><div style='color:#9fb0da'>{c}</div><div style='width:60%'><div style='height:10px; background:rgba(255,255,255,0.04); border-radius:6px; overflow:hidden'><div style='width:{p}%; height:100%; background:{'#8b5cf6' if i==idx else '#475569'}; border-radius:6px'></div></div></div><div style='width:64px; text-align:right; color:#cbd5ff'>{p:.1f}%</div></div>", unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Upload an MRI image on the left to enable analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

# small footer
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#97a4c6'>NeuroScan — research demo • Not for clinical diagnosis</div>", unsafe_allow_html=True)
