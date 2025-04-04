# edge_detection.py
import cv2
import numpy as np

def detect_edges(image, method='canny'):
    """
    Detect edges in the image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        method (str): Method to use ('canny', 'sobel', 'laplacian').
    
    Returns:
        numpy.ndarray: Image with detected edges.
    """
    if method == 'canny':
        return cv2.Canny(image, 100, 200)
    elif method == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.magnitude(sobelx, sobely)
    elif method == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F)
    else:
        return image
