# Multi-Stage Image Enhancement and Analysis Pipeline

This repository contains the code and deployment of our Computer Vision project. The project is designed as a multi-stage image processing pipeline where each stage applies a different computer vision algorithm. The aim is to enhance images while allowing each team member to contribute a unique algorithm.

## Project Overview

The pipeline consists of the following stages:
1. **Preprocessing & Noise Reduction:** Improves image quality by reducing noise using methods like Gaussian Blur.
2. **Edge Detection & Feature Extraction:** Detects edges using algorithms such as Canny, Sobel, or Laplacian filters.
3. **Super-Resolution / Image Reconstruction:** Enhances low-resolution images to high-resolution using methods like bicubic interpolation (placeholder for SRCNN/ESRGAN).
4. **Object Detection & Segmentation:** Applies models (e.g., YOLO) to detect objects and segment images.
5. **Image Compression & Downscaling:** Reduces image file size using compression techniques like JPEG.


## Live App Demo

You can try out the live version of the app by clicking on the link below:

[Live Demo of the App](https://multi-stage-image-enhancement-and-analysis-pipeline.streamlit.app/)


## Repository Contents

- **src/**: Contains individual modules for each processing stage and a pipeline integration module.
- **streamlit_app.py**: The main application file for deployment on Streamlit.
- **docs/report.pdf**: The project report with objective, approaches, results, deployment and GitHub links, screenshots, and team member contributions.
- **video/demo.mp4**: A video demonstration of the project.
- **assets/**: Contains sample images for testing and screenshots of the deployed application.

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/praj-tarun/Multi-Stage-Image-Enhancement-and-Analysis-Pipeline.git
   cd MultiStage_Image_Pipeline
