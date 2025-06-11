import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Load models once
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    detr = pipeline("object-detection", model="facebook/detr-resnet-50")
    blip_base = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    blip_large = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    return yolo, detr, blip_base, blip_large

yolo_model, detr_pipeline, blip_base_pipeline, blip_large_pipeline = load_models()

# Draw bounding boxes for DETR
def draw_boxes_detr(image: Image.Image, detections, threshold=0.3):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except:
        font = ImageFont.load_default()
    for det in detections:
        if det['score'] >= threshold:
            box = det['box']
            label = f"{det['label']} ({det['score']:.2f})"
            draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline="red", width=2)
            draw.text((box['xmin'], box['ymin'] - 10), label, fill="red", font=font)
    return image

# Layout
st.set_page_config(page_title="Visual AI Comparison", layout="wide")
st.title("🤖 Visual AI Comparison: YOLOv8 vs DETR vs BLIP")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    run_button = st.button("🚀 Run All Models")
    if run_button:
        col1, col2, col3 = st.columns(3)

        # Column 1: Input Image
        with col1:
            st.subheader("📥 Input Image")
            st.image(image, use_column_width=True)

        # Column 2: Object Detection Outputs
        with col2:
            st.subheader("🟥 YOLOv8 Detection")
            yolo_results = yolo_model(image)
            yolo_img = yolo_results[0].plot()
            yolo_img_pil = Image.fromarray(yolo_img[..., ::-1])
            st.image(yolo_img_pil, use_column_width=True)

            st.subheader("🟦 DETR Detection")
            detr_results = detr_pipeline(image)
            detr_img = draw_boxes_detr(image.copy(), detr_results)
            st.image(detr_img, use_column_width=True)

        # Column 3: Captions and Summary
        with col3:
            st.subheader("📋 Captions & Detection Summary")

            blip_base_caption = blip_base_pipeline(image)[0]['generated_text']
            blip_large_caption = blip_large_pipeline(image)[0]['generated_text']

            yolo_labels = [yolo_model.names[int(box.cls)] for box in yolo_results[0].boxes] if yolo_results[0].boxes else []
            yolo_summary = ", ".join(set(yolo_labels)) if yolo_labels else "No objects detected."

            detr_text = "\n".join([f"{obj['label']} ({obj['score']:.2f})" for obj in detr_results]) if detr_results else "No objects detected."

            st.markdown(f"### 📝 BLIP Captions")
            st.markdown(f"- **Base**: {blip_base_caption}")
            st.markdown(f"- **Large**: {blip_large_caption}")

            st.markdown("### 🟥 YOLOv8 Detected Objects")
            st.markdown(f"- {yolo_summary}")

            st.markdown("### 🟦 DETR Detected Objects")
            st.markdown(detr_text)
