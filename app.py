import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import json

# LOAD CLASS NAMES
with open("models/classes.json", "r") as f:
    class_names = json.load(f)

# LOAD MODEL
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

model.load_state_dict(torch.load("models/prototype_model.pth", map_location="cpu"))
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1)

    st.success(f"Prediction: {class_names[pred.item()]}")