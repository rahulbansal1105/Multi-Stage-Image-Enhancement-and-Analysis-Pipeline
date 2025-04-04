# preprocessing.py
import cv2

def reduce_noise(image, method='gaussian'):
    """
    Reduce noise in the image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        method (str): Method to use ('gaussian', 'median', 'nlm').
    
    Returns:
        numpy.ndarray: Noise-reduced image.
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'nlm':
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        return image
