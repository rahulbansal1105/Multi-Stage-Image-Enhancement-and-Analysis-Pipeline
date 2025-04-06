# super_resolution.py Raghvendra
import cv2

def super_resolve(image, method='bicubic'):
    """
    Enhance the resolution of an image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        method (str): Method to use ('bicubic' as a placeholder for others like SRCNN or ESRGAN).
    
    Returns:
        numpy.ndarray: High-resolution image.
    """
    if method == 'bicubic':
        height, width = image.shape[:2]
        return cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    else:
        return image
