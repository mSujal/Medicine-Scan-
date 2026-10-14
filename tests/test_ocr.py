import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from src.ocr import read_text

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None,
                                reason="tesseract not installed")


def _render(text: str) -> Image.Image:
    img = Image.new("RGB", (900, 200), "white")
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((20, 50), text, fill="black", font=font)
   
    return img


def test_reads_rendered_text():
    out = read_text(_render("PARACETAMOL 500 mg"), load_config()).lower()