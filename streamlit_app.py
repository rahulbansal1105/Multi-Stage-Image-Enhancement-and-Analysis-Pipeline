import streamlit as st
import cv2
import numpy as np
import graphviz

# Import feature functions from respective modules
from src.preprocessing import reduce_noise
from src.edge_detection import detect_edges
from src.super_resolution import super_resolve
from src.object_detection import detect_objects
from src.image_compression import compress_image

st.set_page_config(page_title="Multi-Stage Image Enhancement Pipeline", layout="wide")

st.title("Multi-Stage Image Enhancement and Analysis Pipeline")

# Image Upload Section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    
    # Create a working copy of the image
    processed_image = original_image.copy()
    
    # Display Original Image
    st.subheader("Original Image")
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    # Sidebar for toggling feature stages
    st.sidebar.header("Toggle Feature Stages")
    # Keep track of applied features (to create a DAG)
    applied_features = []
    
    # Layout for feature controls
    st.sidebar.markdown("Select the features you want to apply on the image in sequence. Each feature will be applied on top of the previous stage.")

    # Feature 1: Preprocessing & Noise Reduction
    if st.sidebar.checkbox("1. Preprocessing & Noise Reduction"):
        processed_image = reduce_noise(processed_image, method='gaussian')
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="After Preprocessing & Noise Reduction", use_column_width=True)
        applied_features.append("Preprocessing")
    
    # Feature 2: Edge Detection & Feature Extraction
    if st.sidebar.checkbox("2. Edge Detection & Feature Extraction"):
        # For edge detection, the output is a grayscale image
        edge_img = detect_edges(processed_image, method='canny')
        st.image(edge_img, caption="After Edge Detection", use_column_width=True)
        # Update processed_image for subsequent stages.
        # For visualization, we convert edges to 3 channels.
        processed_image = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        applied_features.append("Edge Detection")
    
    # Feature 3: Super-Resolution / Image Reconstruction
    if st.sidebar.checkbox("3. Super-Resolution / Image Reconstruction"):
        processed_image = super_resolve(processed_image, method='bicubic')
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="After Super-Resolution", use_column_width=True)
        applied_features.append("Super-Resolution")
    
    # Feature 4: Object Detection & Segmentation
    if st.sidebar.checkbox("4. Object Detection & Segmentation"):
        # detect_objects returns both processed image and bounding boxes
        detected_image, boxes = detect_objects(processed_image, method='yolo')
        # For demonstration, we draw boxes on the image.
        for box in boxes:
            cv2.rectangle(detected_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        processed_image = detected_image.copy()
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="After Object Detection", use_column_width=True)
        applied_features.append("Object Detection")
    
    # Feature 5: Image Compression & Downscaling
    if st.sidebar.checkbox("5. Image Compression & Downscaling"):
        processed_image = compress_image(processed_image, quality=80)
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="After Compression", use_column_width=True)
        applied_features.append("Compression")
    
    # Final Processed Image
    st.header("Final Processed Image")
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Final Image", use_column_width=True)
    
    # Create and display DAG of applied features using Graphviz
    if applied_features:
        dag_source = "digraph G {"
        dag_source += '"Original Image" -> '
        # Join applied features in order
        dag_source += " -> ".join(f'"{feat}"' for feat in applied_features)
        dag_source += "}"
        
        st.subheader("Feature Application DAG")
        st.graphviz_chart(dag_source)
    else:
        st.info("No features applied yet. Use the sidebar to toggle features and see the processing pipeline.")
