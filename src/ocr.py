"""OCR wrapper. Swappable but for now tesseract is used"""
import numpy as np 
import pytesseract
from PIL import Image


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 3: 
        arr = arr[:, :, ::-1]       # Opencv BGR -> RGB
    
    return Image.fromarray(arr)


def read_text(img, cfg: dict):
    """Return raw OCR text for an image (PIL or numpy array)"""
    ocr_cfg = cfg["ocr"]
    engine = ocr_cfg["engine"]
    if engine != "tesseract":
        raise NotImplementedError(f"OCR engine '{engine}' not implemented")
    
    lang = "+".join(ocr_cfg.get("languages", ["eng"]))
    psm = ocr_cfg.get("psm", 11)
    return pytesseract.image_to_string(_to_pil(img), lang=lang, config=f"--psm {psm}")
