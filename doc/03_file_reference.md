# File Reference

What each file is for, and whether it exists yet.

## Project root

| File | Purpose | Status |
|---|---|---|
| `config.yaml` | All tunable settings (OCR, preprocessing, matcher, fusion, thresholds, paths) | Done |
| `requirements.txt` | Python dependencies (needs `tensorflow>=2.16` for Python 3.12; avoid `easyocr`/`torch`) | Check contents |
| `.gitignore` | Keeps the venv, caches, model files, and `data/raw/` (private photos) out of git. No trailing spaces on any line | Done |
| `README.md` | Public project summary. Do not put a link to private raw photos here | Pending |

## `src/` (the pipeline)

| File | Purpose | Status |
|---|---|---|
| `config.py` | `load_config()` reads the YAML. `resolve_path()` turns a `paths` entry into an absolute path | Done |
| `preprocess.py` | `preprocess_for_ocr()`: grayscale, downscale cap, cubic upscale for small images, optional CLAHE and denoise | Done |
| `ocr.py` | `read_text()`: Tesseract through `pytesseract`, one swappable function. Accepts PIL images or arrays | Done |
| `matcher.py` | Text branch. `normalize`, `extract_strengths` (regex), `_tokens` (OCR digit fixes), `_ngrams`, `_score`, `text_scores`, `load_name_index` | Done |
| `embedder.py` | Visual branch. MobileNetV2 embeddings, L2-normalised, lazy-loaded | Done |
| `retriever.py` | Build an embedding index from reference images and score by cosine kNN, max per ingredient | Pending |
| `fusion.py` | Combine text and visual score dicts with config weights | Pending |
| `decision.py` | Top-3, category lookup, strength validation, abstain rules, `Result` dataclass | Pending |
| `pipeline.py` | `predict(images, cfg)` end to end, returns intermediates for failure analysis | Pending |

## `tests/`

| File | What it checks | Last reported |
|---|---|---|
| `test_matcher.py` | Normalisation, strength regex, exact/alias/truncated/OCR-confusion matches, multi-ingredient, no false positives on boilerplate, real reference DB loads | 15 passed |
| `test_ocr.py` | Renders text into an image and checks Tesseract reads it back. Skipped if `tesseract` is not installed | 1 passed |
| `test_embedder.py` | Embedding shape (1280,), unit norm, same image gives similarity 1, different image lower | 2 passed |

Run from the project root with `python -m pytest -q tests/`. Use `python -m pytest`, not bare `pytest`, so `src.` imports resolve.

## `scripts/` (experiments and helpers; run with `python -m scripts.<name>`)

| File | Purpose |
|---|---|
| `__init__.py` | Makes `scripts` a package so `-m` works |
| `ocr_folder.py` | Run OCR and the matcher over every image in a folder and print text, strengths, and matches. Use it to see what Tesseract reads |
| `compare_preprocess.py` | Runs each image under four preprocessing variants (raw, CLAHE, upscale, upscale+CLAHE), with a rotation search (0, 90, 270 degrees, keeping the rotation with the best single score), and prints hits and false positives per variant. Its `EXPECTED` dict holds the ground truth for the online sample images |
| `embed_check.py` | Embeds every image in a folder and prints each image's nearest neighbour. A sanity check of the embedder only |

## `data/`

| Path | Purpose |
|---|---|
| `reference_db.csv` | Ingredient layer: `ingredient,strength,category,aliases` (aliases separated by `\|`). Built from the NLEM Nepal 2021 summary plus rows seen on packs. Rows not from the NLEM are labelled market-only |
| `labels.csv` | Ground truth per image: `id, strip_id, brand, manufacturer, ingredient, strength, category, side, split, source, image_path, label_text`. Unreadable values are `TODO`. Out-of-scope packs use `category=out_of_scope` |
| `sources.csv` | URL and date accessed for every hand-downloaded online image |
| `market_brands.csv` | Planned brand list with source and source type. Unknown values stay `TODO` |
| `raw/online/` | Hand-downloaded online images. For building and testing code only. Git-ignored |
| `raw/real/` | Photos we took ourselves. The only source for reported accuracy. Git-ignored |
| `processed/` | Resized copies (about 1024 px long side) for faster embedding. Originals stay untouched |
| `synthetic/` | Code-augmented and hand-made (GIMP) degraded copies. Never used in validation or test. Hand-made ones go in `synthetic/handmade/` with the operations noted |
| `index.npz` | Saved reference embeddings (generated, git-ignored) |

## `docs/`

| File | Content |
|---|---|
| `01_overview.md` | Goal, scope, stages, safety |
| `02_architecture.md` | Data flow, contracts, design choices, config reference |
| `03_file_reference.md` | This file |
| `04_data_plan.md` | Data collection, splits, augmentation, privacy |
| `05_progress_log.md` | What was done, results so far, decisions, open items |

## Naming conventions

- Real photos: `r001_back_a.jpg`, `r001_front_a.jpg`, `r001_back_b.jpg`, ...
- Bottles: `b001_a.jpg` to `b001_d.jpg` (four angles)
- Online images: lowercase product names, with `_front` / `_back` where split
