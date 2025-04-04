# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from pipeline import process_image

st.title("Multi-Stage Image Enhancement and Analysis Pipeline")
st.write("Upload an image to see each processing stage of the pipeline.")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read and decode the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Process the image through the pipeline
    results = process_image(image)
    
    # Display results for each stage
    st.subheader("Preprocessed (Noise Reduced) Image")
    st.image(cv2.cvtColor(results['noise_reduced'], cv2.COLOR_BGR2RGB), channels="RGB")
    
    st.subheader("Edge Detection Result")
    st.image(results['edges'], use_column_width=True)
    
    st.subheader("Super-Resolution Result")
    st.image(cv2.cvtColor(results['super_resolved'], cv2.COLOR_BGR2RGB), channels="RGB")
    
    st.subheader("Object Detection Result")
    detected_img, boxes = results['object_detection']
    # Draw bounding boxes (for demonstration purposes)
    for box in boxes:
        cv2.rectangle(detected_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), channels="RGB")
    
    st.subheader("Compressed Image")
    st.image(cv2.cvtColor(results['compressed'], cv2.COLOR_BGR2RGB), channels="RGB")
