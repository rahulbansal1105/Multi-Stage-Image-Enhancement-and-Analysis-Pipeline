import streamlit as st
import cv2  # type: ignore
import numpy as np
import graphviz  # type: ignore
import os
from streamlit_sortables import sort_items  # type: ignore

# Import feature functions from respective modules
from src.preprocessing import preprocess_image
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
# Define Feature Categories and Methods
# ----------------------------
feature_categories = {
    "Preprocessing & Noise Reduction": {
        "Gaussian Blur": lambda img: preprocess_image(img, "gaussian"),
        "Bilateral Filter": lambda img: preprocess_image(img, "bilateral"),
        "Median Filter": lambda img: preprocess_image(img, "median"),
        "Sharpening": lambda img: preprocess_image(img, "sharpen"),
        "Denoising": lambda img: preprocess_image(img, "denoise"),
        "Histogram Equalization": lambda img: preprocess_image(img, "hist_eq"),
        "CLAHE": lambda img: preprocess_image(img, "clahe"),
        "Gamma Correction": lambda img: preprocess_image(img, "gamma"),
        "Grayscale Conversion": lambda img: preprocess_image(img, "grayscale"),
        "HSV Conversion": lambda img: preprocess_image(img, "hsv"),
        "Rotation": lambda img: preprocess_image(img, "rotate", angle=90),
        "Flipping": lambda img: preprocess_image(img, "flip", direction=1),
        "Normalization": lambda img: preprocess_image(img, "normalize_minmax"),
    },
    "Edge Detection & Feature Extraction": {
        "Canny Edge Detection": lambda img: cv2.cvtColor(detect_edges(img, method='canny'), cv2.COLOR_GRAY2BGR),
        "Sobel Edge Detection": lambda img: cv2.cvtColor(detect_edges(img, method='sobel'), cv2.COLOR_GRAY2BGR)
    },
    "Super-Resolution / Image Reconstruction": {
        "Bicubic Super-Resolution": lambda img: super_resolve(img, method='bicubic'),
        "Nearest-Neighbor Super-Resolution": lambda img: super_resolve(img, method='nearest_neighbor'),
        "Convolutional Neural Network Super-Resolution": lambda img: super_resolve(img, method='srcnn'),
        "Swin Transformer for Image Restoration": lambda img: super_resolve(img, method='swinir')
    },
    "Object Detection & Segmentation": {
        "SLIC": lambda img: detect_objects(img, method='slic'),
        "YOLO": lambda img: detect_objects(img, method='yolo'),
        "RetinaNet": lambda img: detect_objects(img, method='RetinaNet'),
        #"Contour-Based Segmentation": lambda img: detect_objects(img, method='contour')[0]
    },
    "Image Compression & Downscaling": {
        "JPEG Compression (80%)": lambda img: compress_image(img, quality=80),
        "JPEG Compression (50%)": lambda img: compress_image(img, quality=50)
    }
}

# ----------------------------
# Sidebar: Feature Selection and Ordering
# ----------------------------
st.sidebar.header("Feature Controls")
st.sidebar.markdown("Select methods under each category and reorder them below:")

# Expandable category selection
for category, methods in feature_categories.items():
    with st.sidebar.expander(category, expanded=False):
        for method, function in methods.items():
            col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
            with col1:
                if st.button("➖", key=f"remove_{method}"):
                    if method in st.session_state.selected_features:
                        st.session_state.selected_features.remove(method)
            with col2:
                st.markdown(f"<div style='padding:5px;'>{method}</div>", unsafe_allow_html=True)
            with col3:
                if st.button("➕", key=f"add_{method}"):
                    st.session_state.selected_features.append(method)

# Display and reorder selected features
if st.session_state.selected_features:
    st.header("Applied Features")
    st.session_state.selected_features = sort_items(st.session_state.selected_features)

# ----------------------------
# Image Selection / Upload
# ----------------------------

st.sidebar.subheader("Select or Upload Image")
default_images = {
    "Sample 1": "assets/sample1.jpg",
    "Sample 2": "assets/sample2.jpg",
    "Sample 3": "assets/sample3.jpg"
}

# Radio button to choose sample image or upload
selected_option = st.sidebar.radio("Choose Image", list(default_images.keys()) + ["Upload Your Own"])

if selected_option == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)
    else:
        original_image = cv2.imread(default_images["Sample 1"])  # Default fallback
else:
    original_image = cv2.imread(default_images[selected_option])

# ----------------------------
# Image Processing
# ----------------------------
processed_image = original_image.copy()
for feature in st.session_state.selected_features:
    for category, methods in feature_categories.items():
        if feature in methods:
            processed_image = methods[feature](processed_image)

# ----------------------------
# Display Images (CSS-Based Responsive Sizing)
# ----------------------------

# Inject CSS for responsive image display
st.markdown(
    """
    <style>
        .responsive-img {
            width: 30vw; /* 30% of viewport width */
            height: auto; /* Maintain aspect ratio */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display images side by side
col1, col2 = st.columns([1, 1])
with col1:
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=False, output_format="PNG")
with col2:
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=False, output_format="PNG")

# ----------------------------
# DAG Visualization of Applied Features
# ----------------------------
st.subheader("Feature Application Sequence")
dag = graphviz.Digraph(format="png")
dag.attr(rankdir="LR", size="10")

# Start with original image node
previous_node = "Original Image"
dag.node(previous_node, shape="rect", style="filled", color="lightblue")

# Add nodes for applied features
for feature in st.session_state.selected_features:
    dag.node(feature, shape="rect", style="filled", color="lightgreen")
    dag.edge(previous_node, feature)
    previous_node = feature

# Final processed image node
dag.node("Processed Image", shape="rect", style="filled", color="lightcoral")
dag.edge(previous_node, "Processed Image")

st.graphviz_chart(dag)
