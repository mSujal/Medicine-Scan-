import copy 
import sys 
from pathlib import Path 

from PIL import Image 

from src.config import load_config, resolve_path 
from src.matcher import load_name_index, text_scores 
from src.ocr import read_text 
from src.preprocess import preprocess_for_ocr

EXPECTED = {
    "alernil": {"fexofenadine"},
    "cetamol": {"paracetamol"},
    "medomol": {"paracetamol"},
    "sinex": {"paracetamol"},
    "flexon_back": {"paracetamol", "ibuprofen"},
    "omecap": {"omeprazole"},
    "omepraz": {"omeprazole"},
    "pantafast": {"domperidone"},
    "febuget": set(),
}

VARIANTS = {
    "raw": dict(clahe=False, upscale=False), 
    "clahe": dict(clahe=True, upscale=False ),
    "up": dict(clahe=False, upscale=True),
    "up+clahe": dict(clahe=True, upscale=True),
}

def best_over_rotation(img, cfg, index):
    m = cfg["matcher"]
    best, best_angle = {}, 0
    for angle in (0, 90, 270):
        proc = preprocess_for_ocr(img.rotate(angle, expand=True), cfg)
        text = read_text(proc, cfg)
        s = text_scores(text, index, m["min_ratio"], m["min_len"], m["max_ngram"])
        if sum(s.values()) > sum(best.values()):
            best, best_angle = s, angle 
    
    return best, best_angle 

def main(folder):
    base = load_config() 
    index = load_name_index(resolve_path(base, "reference_db"))
    hits = {v: 0 for v in VARIANTS}
    false_pos = {v: 0 for v in VARIANTS}
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png" ".webp"}:
            continue
        img = Image.open(p).convert("RGB")
        expected = EXPECTED.get(p.stem)
        print(f"=== {p.name} expected={sorted(expected) if expected is not None else '?'}")
        for name, over in VARIANTS.items():
            cfg = copy.deepcopy(base) 
            cfg["preprocess"].update(over)
            found, angle = best_over_rotation(img, cfg, index)
            print(f" {name:9s} rot={angle:3d} {({k: round(v, 2) for k, v in found.items()})}")
            if expected is not None: 
                hits[name] += expected <= set(found) and bool(expected)
                false_pos[name] += len(set(found) - expected)
    print("\nVariant   hits  false_positives")
    for v in VARIANTS:
        print(f"{v:9s} {hits[v]:4d}  {false_pos[v]:4d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/online")