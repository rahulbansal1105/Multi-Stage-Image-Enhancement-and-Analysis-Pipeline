# object_detection.py
import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb
import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO

def detect_objects(image, method='slic'):
    """
    Detect objects in the image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        method (str): Object detection method ('slic'.).
    
    Returns:
        tuple: Processed image and a list of detected bounding boxes.
    """
    if method == 'slic':
        # Dummy bounding box for demonstration purposes
        image_segments=slic(image,n_segments=20,compactness=13)
        final_image=label2rgb(image_segments,image,kind='avg')
        return final_image
    elif method == 'yolo':
        model = YOLO("models/yolov5su.pt") 

        image_segments=slic(image,n_segments=20,compactness=13)
        rgb_image=label2rgb(image_segments,image,kind='avg')
        # image_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
        image_pil = Image.fromarray((image * 255).astype(np.uint8))


        results = model.predict(source=image_pil, conf=0.4, imgsz=640, verbose=False)[0]
        draw = ImageDraw.Draw(image_pil)

        boxes = results.boxes
        if boxes is not None and boxes.data is not None:
            prev_score = 0
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cls_id = int(boxes.cls[i].item())
                score = boxes.conf[i].item()
                if score  > prev_score:
                    prev_score = score
                    label = f"{model.names[cls_id]}: {score:.2f}"

                    y1_label = max(y1 - 20, 0)
                    draw.rectangle([(x1, y1_label), (x2, y2)], outline='red', width=2)
                    draw.text((x1, y1_label), label, fill='red')
        return np.array(image_pil)
    else:
        return image

