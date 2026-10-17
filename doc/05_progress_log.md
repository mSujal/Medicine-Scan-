# Progress Log

Results marked "reported" came from terminal output the team pasted. Items marked "user-confirmed" were checked by the team without the output being shown.

## Environment

- Linux (Arch-based), Python 3.12, virtualenv in `.venv/`
- TensorFlow 2.21.0 imports and runs (reported)
- Tesseract installed with `pacman` (`tesseract`, `tesseract-data-eng`, `tesseract-data-nep`). `pytesseract` and `pillow` installed with pip
- OpenCV: `opencv-python-headless`
- Other packages: `pyyaml`, `pandas`, `rapidfuzz`, `pytest`

## What was done

### 1. Config
`config.yaml` plus `src/config.py`. Early YAML mistakes taught two things: the key must be `languages` (the code ignores `language`), and YAML must use indented block style.

### 2. Reference database (starter)
About 20 rows from the NLEM Nepal 2021 summary, with aliases and categories, plus rows seen on the sample packs (pantoprazole 40 mg, domperidone 30 mg, fexofenadine 180 mg, market-only). Strengths must be verified against the NLEM PDF. The team is now building the real database by hand.

### 3. Matcher and tests
Normalisation, strength regex, OCR digit repair, n-gram fuzzy matching against names and aliases. The first version used `partial_ratio` and was replaced after it produced false positives (see Experiments). Bugs fixed along the way: `tokens(text)` called instead of `_tokens(text)`, `.parent[1]` instead of `.parents[1]`, an empty CSV, and a `/` typed instead of `<` in the prefix-coverage condition of `_score`. 15 tests pass (reported).

### 4. OCR
Tesseract chosen over EasyOCR to keep the environment light (no PyTorch). `read_text()` is a single swappable function. Clean rendered text test passes (reported).

### 5. Preprocessing
`preprocess_for_ocr()` with flags. Bugs fixed: `titleGridSize` should be `tileGridSize`, and `cv.INTER_CUBIC` should be `cv2.INTER_CUBIC`. A downscale cap (`max_long_side`) was added because phone photos are very large.

### 6. Experiments scripts
`ocr_folder.py`, `compare_preprocess.py`, `embed_check.py` (see `03_file_reference.md`).

### 7. Embedder
MobileNetV2 embeddings, 2 tests pass in about 13 seconds (reported).

### 8. Data collection
Pharmacy photos collected (about 45 images, backs, fronts, and bottles from several angles). Labelling and the hand-built reference database are in progress.

### 9. Pending fixes
Applied and passing, user-confirmed: three reference rows, `clahe: false` and `upscale: true`, the new `best_over_rotation` (best single score instead of sum of scores), the lowered `min_ratio` of 80, and the updated `EXPECTED` entries.

## Experiments and results

### A. Raw OCR on 15 online images (reported)
About half gave the right ingredient with no preprocessing (Alernil, Medomol, Omepraz, Flexon back, Cetamol, Pantafast partially). Failures: crumpled or glary foil (Omecap, Sinex), very low resolution (Pantocid, Niko), and bottles with curved labels (Tusq-Dx, Tussinol, Beta-2). Febuget (febuxostat, out of scope) correctly produced no match. A false positive appeared: `ondansetron` at 0.86 on the Alernil pack.

### B. Preprocessing comparison (reported, original matcher, 8 labelled positive images)

| Variant | Hits | False positives |
|---|---|---|
| raw | 5 | 3 |
| clahe | 4 | 5 |
| upscale | 7 | 10 |
| upscale + clahe | 6 | 8 |

Findings:
- Upscaling helped small images (Cetamol, Flexon).
- CLAHE did not help on glossy foil. Probably it amplifies glare. It stays a switchable flag, and default is off.
- Omecap failed under every variant and is logged as a failure case.
- With 8 images, a difference of 1 hit is not a finding.
- Almost every false positive scored 0.86, just over the old 85 cutoff. This led to the matcher scoring change.
- The old rotation search summed scores, so several junk matches beat one real hit. It now keeps the rotation with the best single score.

Note: this comparison was run before the matcher fix. It should be rerun on real photos once `labels.csv` is ready.

### C. Embedder sanity check on about 45 photos (reported)
Nearest neighbours were mostly the same image type (backs with backs, fronts with fronts, bottles with bottles), with cosine similarities between 0.5 and 0.84. Interpretation: the pretrained CNN separates by appearance, not by ingredient. It will be useful for blister vs bottle and front vs back, and it is weak alone for naming the ingredient. This is expected to appear in the final ablation.

## Decisions

| Decision | Reason |
|---|---|
| Tesseract, not EasyOCR | Avoids PyTorch, keeps one deep learning framework |
| Dictionary-constrained matching, no NER | Too little data. Regex and dictionary are simpler and robust |
| Pretrained MobileNetV2 frozen, small head trained | Fits the course (Ch. 9, 12). No backbone fine-tuning |
| Hand fusion weights 0.7/0.3 as a baseline | Can be replaced by logistic regression if time allows |
| Return all ingredients above threshold | Combination products have several actives |
| Bottles: multi-image collection, no reconstruction | Curved labels hide text from one angle. Combine per-image scores |
| Synthetic images for training only | Reported accuracy uses real photos only |
| Default `clahe: false`, `upscale: true` | Measured on the sample set |
| Server-backed Android app first | On-device is future work. Depends on supervisor answer about offline use |

## Current stage

Stage 1 (laptop pipeline) in progress. Done: text branch, preprocessing, embedder, first experiments, photo collection. Next in order: labels and reference database, retriever, fusion, decision layer, `predict()`, Keras head vs kNN vs SVM, evaluation table (text only, image only, fused), then the FastAPI endpoint and Android client.

## Open items to confirm

1. What exactly is the required output: ingredient, category, or brand? (Supervisor.)
2. Must the Android app work offline, or is a server-backed app acceptable? (Supervisor.)
3. Are bottles in scope for the evaluation? (Supervisor.)
4. Does the exam delay change the final presentation date? Which capstone week are we in?
5. Which OCR language setup handles Devanagari best? Test `eng` vs `eng+nep` on real photos.
6. Strength rows in `reference_db.csv` need verification against the NLEM PDF.
7. Aspirin is listed as a fever row only because it is on the complementary list. It needs an age warning in the decision layer.

## Known limitations

- Online-image results prove the code runs, not that the system works on real Nepali strips.
- OCR fails on heavily crumpled foil, tiny images, and some rotated text.
- The image branch alone cannot reliably name an ingredient. Many tablets look alike.
- No GPU assumed. Everything runs on CPU.
