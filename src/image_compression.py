# image_compression.py
import cv2

def compress_image(image, quality=90):
    """
    Compress the image using JPEG compression.
    
    Parameters:
        image (numpy.ndarray): Input image.
        quality (int): JPEG quality parameter (lower means more compression).
    
    Returns:
        numpy.ndarray: Compressed image.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if result:
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    return image
