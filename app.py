import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch

# Load all models from local
@st.cache_resource
def load_models():
    yolo = YOLO("models/yolov8n.pt")

    # Load DETR
    detr_model = DetrForObjectDetection.from_pretrained("models/detr")
    detr_processor = DetrImageProcessor.from_pretrained("models/detr")

    # BLIP base
    blip_base_model = BlipForConditionalGeneration.from_pretrained("models/blip-base")
    blip_base_processor = BlipProcessor.from_pretrained("models/blip-base")

    # BLIP large
    blip_large_model = BlipForConditionalGeneration.from_pretrained("models/blip-large")
    blip_large_processor = BlipProcessor.from_pretrained("models/blip-large")

    return yolo, (detr_processor, detr_model), (blip_base_processor, blip_base_model), (blip_large_processor, blip_large_model)

yolo_model, (detr_processor, detr_model), (blip_base_proc, blip_base_model), (blip_large_proc, blip_large_model) = load_models()

# Draw boxes from DETR
def draw_boxes_detr(image: Image.Image, outputs, threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > threshold:
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]-10), f"{label} ({score:.2f})", fill="red", font=font)

    return image

# Generate caption
def generate_caption(processor, model, image):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# UI Layout
st.set_page_config(page_title="Visual AI Comparison", layout="wide")
st.title("🤖 Visual AI Comparison: YOLOv8 vs DETR vs BLIP")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    run_button = st.button("🚀 Run All Models")
    
    if run_button:
        col1, col2, col3 = st.columns(3)

        # Column 1: Input
        with col1:
            st.subheader("📥 Input Image")
            st.image(image, use_column_width=True)

        # Column 2: Object Detection
        with col2:
            st.subheader("🟥 YOLOv8 Detection")
            yolo_results = yolo_model(image)
            yolo_img = yolo_results[0].plot()
            yolo_pil = Image.fromarray(yolo_img[..., ::-1])
            st.image(yolo_pil, use_column_width=True)

            st.subheader("🟦 DETR Detection")
            detr_inputs = detr_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = detr_model(**detr_inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

            detr_boxes = {
                'boxes': results["boxes"],
                'scores': results["scores"],
                'labels': [detr_model.config.id2label[int(i)] for i in results["labels"]]
            }
            detr_image = draw_boxes_detr(image.copy(), detr_boxes)
            st.image(detr_image, use_column_width=True)

        # Column 3: Captions
        with col3:
            st.subheader("📋 Captions & Summary")

            base_caption = generate_caption(blip_base_proc, blip_base_model, image)
            large_caption = generate_caption(blip_large_proc, blip_large_model, image)

            yolo_labels = [yolo_model.names[int(cls)] for cls in yolo_results[0].boxes.cls] if yolo_results[0].boxes else []
            yolo_summary = ", ".join(set(yolo_labels)) if yolo_labels else "No objects detected."

            detr_summary = "\n".join([
                f"{label} ({score:.2f})" 
                for label, score in zip(detr_boxes['labels'], detr_boxes['scores'])
            ]) or "No objects detected."

            st.markdown(f"### 📝 BLIP Captions")
            st.markdown(f"- **Base**: {base_caption}")
            st.markdown(f"- **Large**: {large_caption}")

            st.markdown("### 🟥 YOLOv8 Detected Objects")
            st.markdown(f"- {yolo_summary}")

            st.markdown("### 🟦 DETR Detected Objects")
            st.markdown(detr_summary)
