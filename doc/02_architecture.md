# Architecture

## Data flow

```
front image(s)         back image(s)
    |                      |
    v                      v
[1 Preprocess]        [1 Preprocess]
 (resize for           (cap huge sizes, upscale small,
  embedder only)        optional contrast tools)
    |                      |
    v                      v
[4 Embedder]          [2 OCR] -> raw text
 MobileNetV2           |
 -> 1280-d vector      v
    |                 [3 Matcher]
    v                  normalise -> OCR digit fixes ->
[5 Retriever]          fuzzy match n-grams vs ingredient
 cosine kNN vs         names + aliases -> {ingredient: score}
 reference embeddings        |
 -> {ingredient: score}      |
    |                        |
    +-----------+------------+
                v
          [6 Fusion]   weighted sum, weights from config
                v
          [7 Decision] top-3, category lookup, abstain if
                       top < threshold or margin too small
                v
          [8 Output]   ingredient(s), strength, category,
                       confidence, "verify with a pharmacist",
                       debug info
```

## Design rules

1. **Every branch returns the same type:** `dict[str, float]`, ingredient to a score in 0..1. This lets us run text-only, image-only, or fused without changing other code, which is what the ablation needs.
2. **The visual branch is retrieval, not a trained classifier.** Adding a medicine means adding reference images. A small Keras head is trained on top of frozen embeddings for comparison.
3. **The text branch is dictionary-constrained.** Fuzzy matching runs only against the known ingredient and alias list, so a truncated read resolves to a known ingredient instead of a free-form guess.
4. **Fusion handles a missing side.** If a branch has no signal, the other branch is used alone (weights renormalised).
5. **Abstain over guess.** Thresholds live in config and are calibrated on a validation set.
6. **Everything tunable lives in `config.yaml`.**
7. **`predict()` is a pure function:** images in, JSON-friendly data out. No UI or file-path logic inside it. This keeps the later FastAPI/Android step cheap.
8. **Multi-image input.** `predict()` will accept a list of images. Per-ingredient scores from several images are combined (max or noisy-OR). Bottles are just more images.
9. **Portability.** Standard Keras layers only, plain-matrix retrieval with cosine similarity, dependency-light matcher, so a later TFLite/Kotlin port is feasible.

## Module contracts

```python
# src/config.py
def load_config(path="config.yaml") -> dict
def resolve_path(cfg, key) -> Path

# src/preprocess.py
def preprocess_for_ocr(img, cfg) -> np.ndarray       # PIL in, grayscale array out

# src/ocr.py
def read_text(img, cfg) -> str                        # swappable engine (tesseract now)

# src/matcher.py
def normalize(text) -> str
def extract_strengths(text) -> list[str]
def text_scores(text, ref_db, min_ratio, min_len, max_ngram) -> dict[str, float]
def load_name_index(csv_path) -> dict[str, list[str]]

# src/embedder.py
def embed(img) -> np.ndarray                          # (1280,), unit length
def embed_batch(imgs) -> np.ndarray                   # (N, 1280)

# planned
# src/retriever.py   build_index(...), visual_scores(vec, index, k)
# src/fusion.py      fuse(text_s, visual_s, w_text, w_img)
# src/decision.py    decide(scores, ref_db, cfg) -> Result
# src/pipeline.py    predict(images, cfg) -> Result
```

## Key technical choices and why

**Tesseract instead of EasyOCR.** EasyOCR pulls in PyTorch, which would put two deep learning frameworks in one environment alongside TensorFlow. Tesseract is a small system program with a thin Python wrapper. The `read_text()` wrapper means the engine can be swapped later.

**Matcher scoring (`_score`).** The first version used `partial_ratio`, which scores any similar substring as 100. That caused false positives (for example, "ondansetron" matching boilerplate on a fexofenadine pack) and made a truncated read score the same as a perfect one. The current version:
- compares whole strings (`fuzz.ratio`),
- gives truncated prefixes of at least 5 letters and 50% coverage a score between 0.80 and 0.90 (for example "parace" vs "paracetamol" is about 0.81),
- gives exact reads 1.0.

**OCR digit fixes.** `0 -> o`, `1 -> l`, `5 -> s`, applied only to tokens that contain letters, so "paracetamo1" and "IBUPR0FEN" still resolve.

**Preprocessing.** Large phone photos (some 2500 px or more) are downscaled to a cap before OCR to save time and RAM. Small images are upscaled with cubic interpolation. Never a GAN upscaler. CLAHE is a switchable flag, because measurements showed it did not help on glossy foil.

**Embedder.** MobileNetV2 (ImageNet weights, no top, global average pooling) gives a 1280-d vector, L2-normalised, so cosine similarity is a dot product. The model is loaded lazily on first use so that code that never embeds does not pay the TensorFlow import time.

**Combination products.** Packs such as Sinex (chlorpheniramine, phenylephrine, paracetamol), Flexon (ibuprofen and paracetamol), and Pantafast-D (pantoprazole and domperidone) have several actives. The decision layer returns every ingredient above the threshold rather than requiring a single winner.

## Configuration reference (`config.yaml`)

| Key | Meaning |
|---|---|
| `ocr.engine`, `ocr.languages`, `ocr.psm` | OCR engine, Tesseract language codes, page segmentation mode (11 = sparse text) |
| `preprocess.clahe`, `clip_limit` | Contrast tool on/off and strength (currently off) |
| `preprocess.upscale`, `min_long_side` | Upscale images smaller than this long side (max 3x) |
| `preprocess.max_long_side` | Downscale images larger than this (value is a guess, to be tuned) |
| `preprocess.denoise`, `deskew` | Denoise flag; deskew not implemented yet (rotation search is used instead) |
| `embedder.name`, `input_size` | Backbone name, input size 224 |
| `fusion.w_text`, `w_img` | Fusion weights (0.7 / 0.3 baseline, to be tuned on validation data) |
| `decision.min_score`, `min_margin` | Abstain thresholds (placeholders until calibrated) |
| `matcher.min_ratio`, `min_len`, `max_ngram` | Minimum score (x100), minimum n-gram length, longest n-gram |
| `paths.reference_db`, `paths.index` | Reference CSV and embedding index paths |

Function default arguments (for example in `text_scores`) are fallbacks for tests. The pipeline passes the values from `config.yaml`.
