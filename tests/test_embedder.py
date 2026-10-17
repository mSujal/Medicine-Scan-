import numpy as np
from PIL import Image

from src.embedder import embed, embed_batch


def _img(color):
    return Image.new("RGB", (300, 200), color)

def test_embedding_shape_and_unit_norm():
    v = embed(_img("red"))
    assert v.shape == (1280, )
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4

def test_same_image_has_similarity_one_and_different_is_lower():
    a, b = embed_batch([_img("red"), _img("blue")])
    assert abs(float(a @ a) - 1.0) < 1e-4
    assert float(a @ b) < 0.999
