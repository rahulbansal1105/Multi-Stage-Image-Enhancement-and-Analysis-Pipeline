# pipeline.py
from preprocessing import reduce_noise
from edge_detection import detect_edges
from super_resolution import super_resolve
from object_detection import detect_objects
from image_compression import compress_image

def process_image(image):
    """
    Process an image through multiple stages.
    
    Returns:
        dict: A dictionary containing results from each stage.
    """
    # Stage 1: Preprocessing & Noise Reduction
    noise_reduced = reduce_noise(image, method='gaussian')
    
    # Stage 2: Edge Detection & Feature Extraction
    edges = detect_edges(noise_reduced, method='canny')
    
    # Stage 3: Super-Resolution / Image Reconstruction
    super_resolved = super_resolve(noise_reduced, method='bicubic')
    
    # Stage 4: Object Detection & Segmentation
    detected_img, boxes = detect_objects(noise_reduced, method='yolo')
    
    # Stage 5: Image Compression & Downscaling
    compressed = compress_image(noise_reduced, quality=80)
    
    return {
        'noise_reduced': noise_reduced,
        'edges': edges,
        'super_resolved': super_resolved,
        'object_detection': (detected_img, boxes),
        'compressed': compressed
    }
