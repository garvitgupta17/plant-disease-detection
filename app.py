import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import json
import os
import gdown

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# =========================
# LOAD MODEL FILE (DOWNLOAD IF NEEDED)
# =========================
os.makedirs("models", exist_ok=True)

if not os.path.exists("models/prototype_model.pth"):
    url = "PASTE_YOUR_GOOGLE_DRIVE_LINK"
    gdown.download(url, "models/prototype_model.pth", quiet=False)

# =========================
# LOAD CLASSES
# =========================
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_healthy"
]

# =========================
# LOAD MODEL
# =========================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("models/prototype_model.pth", map_location="cpu"))
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# UI DESIGN
# =========================
st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a leaf image and detect disease instantly</p>", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    with col2:
        st.success(f"🌱 Prediction: {class_names[pred.item()]}")
        st.info(f"📊 Confidence: {confidence.item()*100:.2f}%")

    st.progress(int(confidence.item()*100))

st.divider()
st.caption("Built using PyTorch + Streamlit")
