import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import os
import numpy as np
import tempfile

st.set_page_config(page_title="Image Analysis: YOLO vs BLIP", layout="wide")

@st.cache_resource
def load_models():
    """Load YOLO and BLIP models"""
    models = {}
    
    # Use temporary directory for model storage in cloud
    temp_dir = tempfile.gettempdir()
    
    # BLIP Base Model
    blip_dir = os.path.join(temp_dir, "blip-image-captioning-base")
    try:
        if os.path.exists(blip_dir) and os.listdir(blip_dir):
            blip_processor = BlipProcessor.from_pretrained(blip_dir, local_files_only=True)
            blip_model = BlipForConditionalGeneration.from_pretrained(blip_dir, local_files_only=True)
        else:
            st.info("Loading BLIP model...")
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            os.makedirs(blip_dir, exist_ok=True)
            blip_processor.save_pretrained(blip_dir)
            blip_model.save_pretrained(blip_dir)
        
        models['blip'] = (blip_processor, blip_model)
    except Exception as e:
        st.error(f"Error loading BLIP model: {e}")
        return None
    
    # YOLOv11 Model
    try:
        st.info("Loading YOLOv11 model...")
        yolo_model = YOLO('yolo11n.pt')  # Will download if not exists
        models['yolo'] = yolo_model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None
    
    return models

def generate_blip_caption(image, processor, model):
    """Generate BLIP caption"""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=4)
    return processor.decode(output[0], skip_special_tokens=True)

def detect_yolo_objects(image, model, conf_threshold=0.5):
    """YOLOv11 object detection"""
    results = model(image, conf=conf_threshold)
    detected_objects = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detected_objects.append({
                    'label': result.names[int(box.cls)],
                    'confidence': round(float(box.conf), 3),
                    'box': box.xyxy[0].tolist()
                })
    
    return detected_objects

def draw_boxes(image, objects, color='red'):
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for obj in objects:
        box = obj['box']
        label = f"{obj['label']} ({obj['confidence']})"
        
        draw.rectangle(box, outline=color, width=2)
        
        # Text background
        text_bbox = draw.textbbox((box[0], box[1]-20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]-20), label, fill='white', font=font)
    
    return img_copy

def main():
    st.title("Image Analysis: Object Detection vs Image Captioning")
    
    # Load models
    with st.spinner("Loading models... This may take a few minutes on first run."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # File uploader and controls
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Controls section
        st.subheader("Settings")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        
        with col_ctrl2:
            st.write("")  # Spacing
        
        with col_ctrl3:
            analyze_button = st.button("🚀 Analyze Image", type="primary", use_container_width=True)
        
        # Results section - Two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Input Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.header("YOLOv11 Detection")
            if analyze_button:
                with st.spinner("Running YOLOv11..."):
                    yolo_objects = detect_yolo_objects(image, models['yolo'], conf_threshold)
                    yolo_img = draw_boxes(image, yolo_objects, 'red')
                    st.image(yolo_img, use_column_width=True)
                    
                    st.subheader("Detected Objects:")
                    for i, obj in enumerate(yolo_objects, 1):
                        st.write(f"{i}. **{obj['label']}** ({obj['confidence']})")
            else:
                st.info("Click 'Analyze Image' to run detection")
        
        # BLIP Caption section (full width)
        st.header("BLIP Image Caption")
        if analyze_button:
            with st.spinner("Generating caption..."):
                caption = generate_blip_caption(image, models['blip'][0], models['blip'][1])
                st.success(f"**{caption}**")
        else:
            st.info("Click 'Analyze Image' to generate caption")

if __name__ == "__main__":
    main()
