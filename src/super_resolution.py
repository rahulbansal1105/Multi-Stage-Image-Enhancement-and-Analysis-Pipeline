# super_resolution.py Raghvendra

import argparse
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.network_swinir import SwinIR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)  


def run_bicubic(image, scale=4):
    width, height = image.size
    new_size = (width * scale, height * scale)
    return np.array(image.resize(new_size, Image.BICUBIC))


def run_nearest_neighbor(image):
    image_np = np.array(image)
    upscaled_img =  cv2.resize(image_np, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    upscaled_img_pil = Image.fromarray(cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB))
    return np.array(upscaled_img_pil)


def run_srcnn(image: Image.Image) -> Image.Image:
    class SRCNN(nn.Module):
        def __init__(self):
            super(SRCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.conv3(x)
            return x
    
    model = SRCNN().to(device)
    model_path = os.path.join(project_root, 'weights/srcnn_x4.pth')
    model.load_state_dict(torch.load( model_path, map_location=device))
    model.eval()

    image_ycbcr = image.convert('YCbCr')
    y, cb, cr = image_ycbcr.split()

    img_to_tensor = transforms.ToTensor()
    input_tensor = img_to_tensor(y).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_tensor)

    out_img_y = out.squeeze(0).clamp(0, 1).cpu()
    out_img_y = transforms.ToPILImage()(out_img_y)

    final_img = Image.merge("YCbCr", [out_img_y, cb.resize(out_img_y.size), cr.resize(out_img_y.size)]).convert("RGB")
    return np.array(final_img)


def run_swinir(image):
    model = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv'
    )
    
    model = model.to(device)
    image_pp = preprocess_image(image)
    with torch.no_grad():
        image_pp = model(image_pp) 
        image = run_bicubic(image)
    return postprocess_image(image)


def super_resolve(image, method='bicubic'):
    image =  Image.fromarray(image)
    super_res_methods = {
        "bicubic": run_bicubic,
        "nearest_neighbor": run_nearest_neighbor,
        "srcnn": run_srcnn,
        "swinir": run_swinir
    }
    return super_res_methods[method](image)

def preprocess_image(image):
    transform = ToTensor()
    return transform(image).unsqueeze(0).float()

def postprocess_image(tensor_image):
    try:
        image = tensor_image.squeeze().permute(1, 2, 0).numpy() 
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    except Exception: pass
    return tensor_image

def load_image(image_path): 
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def save_image(image, output_path):
    Image.fromarray(image).save(output_path)
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Super Resolution CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--method", type=str, default="bicubic", choices=["bicubic", "srcnn", "nearest_neighbor", "swinir"], help="Super resolution method")
    parser.add_argument("--output", type=str, help="Path to save output image (optional)")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)         
    input_image_path = os.path.join(project_root, args.input)
    input_image = load_image(input_image_path)
    output_image = super_resolve(input_image, method=args.method)

    if args.output: output_path = args.output
    else:
        name, ext = os.path.splitext(args.input)
        output_path = f"{name}_{args.method}_sr{ext}"

    save_image(output_image, output_path)
    
