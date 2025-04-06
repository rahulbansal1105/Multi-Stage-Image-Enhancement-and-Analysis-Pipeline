import cv2
import numpy as np

# ------------------------------------------------
# 1. Custom Denoising Function
# ------------------------------------------------
def custom_denoise(image, strength=0.5):
    """
    Custom denoising using a weighted average of the original and blurred images.
    - `strength`: Controls the level of denoising (0 = no denoising, 1 = full smoothing).
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)  # Apply slight blur
    denoised = cv2.addWeighted(image, 1 - strength, blurred, strength, 0)  # Blend images
    return denoised

# ------------------------------------------------
# 2. Noise Reduction & Smoothing
# ------------------------------------------------
def gaussian_filtering(image, ksize=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image, ksize, sigmaX)

def bilateral_filtering(image, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def median_filtering(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def non_local_means(image, h=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)

def sharpen_image(image, strength=1.5):
    """Applies Unsharp Masking to sharpen the image.
    
    Args:
        image (numpy.ndarray): Input image.
        strength (float): Strength of sharpening effect (default: 1.5).
    
    Returns:
        numpy.ndarray: Sharpened image.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9 * strength, -1],
                       [-1, -1, -1]])
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# ------------------------------------------------
# 3. Contrast Enhancement
# ------------------------------------------------
def histogram_equalization(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def clahe_equalization(image, clipLimit=2.0, tileGridSize=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ------------------------------------------------
# 4. Color Space Transformations
# ------------------------------------------------
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ------------------------------------------------
# 5. Image Transformations (Rotation & Flipping)
# ------------------------------------------------
def rotate_image(image, angle=90):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image, direction=1):
    return cv2.flip(image, direction)

# ------------------------------------------------
# 6. Image Normalization & Scaling
# ------------------------------------------------
def normalize_minmax(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def normalize_zscore(image):
    mean, std = cv2.meanStdDev(image)
    return ((image - mean) / std).astype(np.float32)

# ------------------------------------------------
# Wrapper: Generalized Preprocessing Function
# ------------------------------------------------
def preprocess_image(image, method, **kwargs):
    processing_methods = {
        'gaussian': gaussian_filtering,
        'bilateral': bilateral_filtering,
        'median': median_filtering,
        'nlm': non_local_means,
        'denoise': custom_denoise,  
        "sharpen": lambda img: sharpen_image(img),
        'hist_eq': histogram_equalization,
        'clahe': clahe_equalization,
        'gamma': gamma_correction,
        'grayscale': convert_to_grayscale,
        'hsv': convert_to_hsv,
        'rotate': rotate_image,
        'flip': flip_image,
        'normalize_minmax': normalize_minmax,
        'normalize_zscore': normalize_zscore
    }
    if method in processing_methods:
        return processing_methods[method](image, **kwargs)
    return image  # Return unchanged if method not found

# ------------------------------------------------
# Example Usage (Testing Only)
# ------------------------------------------------
if __name__ == "__main__":
    image = cv2.imread("sample.jpg")
    processed = preprocess_image(image, method='denoise', strength=0.6)  # Example usage
    cv2.imshow("Denoised Image", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
