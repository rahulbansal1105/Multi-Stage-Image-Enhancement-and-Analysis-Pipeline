import streamlit as st
import cv2
import numpy as np
from streamlit_sortables import sort_items

# Import your feature functions from the respective modules
from src.preprocessing import reduce_noise
from src.edge_detection import detect_edges
from src.super_resolution import super_resolve
from src.object_detection import detect_objects
from src.image_compression import compress_image

# Set page configuration for wide layout
st.set_page_config(page_title="Multi-Stage CV Pipeline", layout="wide")

# Title of the application
st.title("Multi-Stage Image Enhancement and Analysis Pipeline")

# Initialize session state for selected features if not already present
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []

# ----------------------------
# Define Available Features and Their Processing Functions
# ----------------------------
feature_funcs = {
    "Preprocessing & Noise Reduction": lambda img: reduce_noise(img, method='gaussian'),
    "Edge Detection & Feature Extraction": lambda img: cv2.cvtColor(detect_edges(img, method='canny'), cv2.COLOR_GRAY2BGR),
    "Super-Resolution / Image Reconstruction": lambda img: super_resolve(img, method='bicubic'),
    "Object Detection & Segmentation": lambda img: detect_objects(img, method='yolo')[0],
    "Image Compression & Downscaling": lambda img: compress_image(img, quality=80)
}

# ----------------------------
# Sidebar: Feature Selection and Ordering
# ----------------------------
st.sidebar.header("Feature Controls")
st.sidebar.markdown("Add or remove features to apply and reorder them by dragging:")

# Display feature options with '+' and '-' buttons
for feature in feature_funcs.keys():
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        st.write(feature)
    with col2:
        if st.button(f"+", key=f"add_{feature}"):
            st.session_state.selected_features.append(feature)
        if st.button(f"-", key=f"remove_{feature}"):
            if feature in st.session_state.selected_features:
                st.session_state.selected_features.remove(feature)

# Display and reorder selected features
if st.session_state.selected_features:
    st.sidebar.subheader("Reorder Features")
    st.session_state.selected_features = sort_items(st.session_state.selected_features)

# ----------------------------
# Main Layout: Image Upload and Display
# ----------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Read the uploaded image (OpenCV loads images in BGR format)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    # Process the image by applying selected features in order
    processed_image = original_image.copy()
    for feature in st.session_state.selected_features:
        processed_image = feature_funcs[feature](processed_image)

    # Display the processed image on the left side with 50% width and padding
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
else:
    st.info("Please upload an image to start.")
