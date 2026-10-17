""" Visual branch
    image -> 1280 dim L2 normalized embedding (pretrained MobileNetV2)
"""

import numpy as np
from PIL import Image

_model = None

def _get_model():
    global _model
    if _model is None:
        import tensorflow as tf     # lazy import
        _model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3)
        )

    return _model


def preprocess_for_embedder(img: Image.Image, size: int = 224):
    """PIL image -> float32 (size, size, 3), scaled to [-1, 1] as required by MobileNetV2"""
    arr = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32)

    return arr / 127.5 - 1.0


def embed(img: Image.Image):
    return embed_batch([img])[0]


def embed_batch(imgs: list):
    """List of PIL images -> (N, 1280) array of unit-length vectors"""
    batch = np.stack([preprocess_for_embedder(i) for i in imgs])
    vecs = _get_model().predict(batch, verbose=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)
