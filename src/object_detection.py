# object_detection.py
import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb
import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

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
    elif method == 'RetinaNet':
        model = retinanet_resnet50_fpn(pretrained=True, weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        model.eval() 

        image_segments=slic(image,n_segments=20,compactness=13)
        rgb_image=label2rgb(image_segments,image,kind='avg')
        image_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))


        image_tensor = F.to_tensor(image_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)[0]

        # Draw bounding boxes
        draw = ImageDraw.Draw(image_pil)
        threshold = 0.2

        COCO_INSTANCE_CATEGORY_NAMES = [ 
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
            if score > threshold:
                box = box.tolist()
                label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
                draw.rectangle(box, outline='red', width=2)
                draw.text((box[0], box[1]), label_text, fill='red')
        return np.array(image_pil)
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

