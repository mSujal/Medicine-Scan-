import cv2 
import numpy as np 

def preprocess_for_ocr(img, cfg: dict):
    """PIL image -> grayscale uint8 array with upscale, CLAHE, denoise"""
    p = cfg["preprocess"]
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if p.get("upscale"):
        long_side = max(gray.shape)
        if long_side < p["min_long_side"]:
            scale = min(3.0, p["min_long_side"] / long_side)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if p.get("clahe"):
        gray = cv2.createCLAHE(clipLimit=p["clip_limit"], tileGridSize=(8, 8)).apply(gray)
    if p.get("denoise"):
        gray = cv2.fastNlMeansDenoising(gray, None, 10)
    
    return gray