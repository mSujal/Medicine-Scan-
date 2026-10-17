# Data Plan

## Data sources

1. **Own photos** (most valuable): pharmacy strips and bottles, photographed by the team. Used for reference, validation, and test.
2. **Friends and family strips**: used and damaged packs. These form the real validation and test sets.
3. **Hand-downloaded online images**: a few dozen, each logged in `data/sources.csv` with URL and date. No crawling. For building and testing code only. They are cleaner than real photos, so they are never trusted for reported accuracy. Check licences, and credit sources if images appear in slides.
4. **Reference text data**: the NLEM Nepal 2021 summary (ingredient and strength layer only), plus market sources for items the NLEM does not cover (cough, antacids). Market sources: DDA Nepal, pharmacy catalogues, Nepali drug-utilisation studies.

## What the NLEM covers and does not

- Covers: ingredient and strength layer for essential medicines in public facilities.
- Does **not** cover: brands, manufacturers, or a home OTC list. Many items are Rx-only.
- Cough is weakly covered (dextromethorphan, ambroxol, chlorpheniramine and phenylephrine are not on the list). Cough rows must come from the market and be labelled "market-only, not NLEM".
- Removed in 2021: cetirizine (replaced by levocetirizine) and the aluminium/magnesium hydroxide antacid tablet. Source antacids from the market.
- Ranitidine is listed, but check whether it is still sold in Nepal.
- Every strength in `reference_db.csv` must be verified against the NLEM PDF or a cited market source. Values written from memory or from a summary are not final.

## Collection protocol (photos)

- Ask permission first. Photograph packs only. No customers, prescriptions, receipts, or pharmacy stickers in frame.
- Blisters: back (printed foil) and front (pill side), 2 to 3 shots each, varied lighting and angle, **no flash** (it creates glare over foil text).
- Include deliberately messy cases: cut strips with 2 to 3 tablets, crumpled foil, hand-held strips.
- Bottles: 3 to 5 photos while rotating the bottle (front, left, right, back).
- Record brand, generic name, strength, and category while the pack is in hand.
- Strip EXIF location data before sharing. Phone photos carry GPS coordinates.
- Keep an offline backup. Treat Drive as a transfer point, not the master copy.

## Definition of "front"

The pill side (clear plastic) is the front, since that is what the CNN sees. The printed foil is the back, which OCR reads.

## Splits (decided per strip, before any augmentation)

| Split | Contents | Purpose |
|---|---|---|
| `ref` / train | Pharmacy strips plus their augmented copies | Reference index, training the small Keras head |
| `val` | About one third of the friends/family strips | Choose thresholds and fusion weights |
| `test` | The rest of the friends/family strips | Frozen before any tuning. Reported once |

Rules:
- Hold out **photos of known strips**, not whole brands.
- All degraded copies of a strip stay on the same side as the original.
- Never tune on the test set.
- Reported accuracy uses **real photos only**. Synthetic accuracy may be shown as a separate, clearly labelled line.

## Augmentation

**Code augmentation** (planned `scripts/augment.py`): rotation (small angles and 90 degree steps), perspective tilt, brightness and contrast changes, Gaussian and motion blur, JPEG compression, noise, fake glare (soft white gradient), fading, and random crops that cut off part of the strip. Settings are recorded so the report can describe them exactly.

**Hand-made scratches** (GIMP): about 10% of each category, training side only. Saved in `data/synthetic/handmade/`, with the operations noted.

**Real damage first.** Collect actual worn and damaged foils from friends and family. Synthetic damage is the last resort.

## Image size handling

Phone photos can be 2500 px or more on the long side.
- OCR input is capped at `preprocess.max_long_side` (a guess, to be tuned) to save time and RAM.
- The embedder resizes everything to 224 px, so processed copies of about 1024 px are stored in `data/processed/` and loaded for embedding.
- Downscaling can blur small print. Test the cap values (for example 2400 vs 3000) with `compare_preprocess` once real photos are labelled.
- Embed in small batches (`batch_size=8`) if memory is low. TensorFlow printed "allocation exceeds 10% of free system memory" warnings when embedding about 45 images at once. These are warnings, not errors.

## Preprocessing experiments to run

Each tool is added as a config flag, compared with `compare_preprocess`, and kept only if it beats upscale-only. None of these should invent text.

- Contrast and sharpening levels (CLAHE clip limits, unsharp masking)
- Adaptive thresholding (Otsu, Sauvola)
- Morphological top-hat or blackhat to flatten foil glare
- Bilateral filtering, gamma correction
- Inverted image (embossed foil text can be light on dark)

## Scope notes for the data

- Combination products have several correct ingredients in `labels.csv`.
- Packs outside the five categories are labelled `out_of_scope` and act as abstain tests.
- Bottles are in scope for the extension stage, using multi-image input.
