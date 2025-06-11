import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from transformers import pipeline

st.title("Model Test")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Load models (minimal)
    yolo = YOLO("yolov8n.pt")
    yolo_results = yolo(img)
    st.image(yolo_results[0].plot(), caption="YOLOv8 Output")

    # Test HF pipeline
    caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = caption_pipe(img)[0]['generated_text']
    st.markdown(f"**Caption:** {caption}")
