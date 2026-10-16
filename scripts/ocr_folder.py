import sys 
from pathlib import Path

from PIL import Image 

from src.config import load_config, resolve_path
from src.matcher import extract_strengths, load_name_index, text_scores
from src.ocr import read_text 

def main(folder: str):
    cfg = load_config()
    index = load_name_index(resolve_path(cfg, "reference_db"))
    m = cfg["matcher"]
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        text = read_text(Image.open(p).convert("RGB"), cfg)
        scores = text_scores(text, index, m["min_ratio"], m["min_len"], m["max_ngram"])
        print(f"=== {p.name}")
        print(" text:", " ".join(text.split())[:200])
        print(" strengths:", extract_strengths(text))
        print(" matches:", scores)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/online")
    