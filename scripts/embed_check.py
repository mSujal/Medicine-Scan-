import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.embedder import embed_batch

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main(folder):
    paths = [p for p in sorted(Path(folder).glob("*")) if p.suffix.lower() in EXTS]
    vecs = embed_batch([Image.open(p).convert("RGB") for p in paths])
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    for i, p in enumerate(paths):
        j = int(sims[i].argmax())
        print(f"{p.name:22s} -> {paths[j].name:22s} cos={sims[i, j]:.2f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/photographed")
