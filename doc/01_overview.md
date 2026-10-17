# MediScan: Project Overview

## What this project is

MediScan is a capstone project for the Samsung Innovation Campus AI course. A user photographs a medicine pack (a blister strip or a bottle). The system outputs:

- the likely **active ingredient(s)**
- the **use category** (fever, cough, cold, gastric, allergy)
- a **confidence score**

If it is not sure, it **abstains** ("unrecognized, please verify") instead of guessing.

**It is an identification aid, never medical advice.** Every result should be shown with a "verify with a pharmacist" message.

## Why it exists

In Nepal, people often hold loose strips with no box, small print, and worn foil. Elderly or non-technical users cannot easily tell what a strip is. The goal is a tool that is usable by non-technical people on an Android phone.

## Grading context

- Capstone is worth 60% of the course grade: quality of the final product, presentation, teamwork.
- The course covers NumPy/Pandas, statistics, KNN/SVM/Naive Bayes, cosine similarity, text preprocessing, Keras CNNs, pretrained models, and design thinking. The project is built to use these.

## Scope

**In scope**
- Categories for home use: fever, cough, cold, gastric, allergy.
- Blister strips (front = pill side, back = printed foil) and bottles (several photos while rotating the bottle).
- Combination products: every ingredient above the threshold is returned, not a single winner.
- A laptop pipeline first (Stage 1), then a server-backed Android app (Stage 2).

**Deliberately out of scope**
- Reading brand names reliably (brand is a bonus, not the target).
- 3D reconstruction or label unwrapping for bottles (multi-image collection only).
- Fine-tuning the pretrained backbone, GAN-based upscaling (can hallucinate characters), NER models, metric learning.
- On-device Android inference (future work).
- Controlled or opioid medicines.
- Out-of-scope drugs (for example febuxostat, amoxicillin) are kept as **deliberate abstain tests**.

## Staged plan

| Stage | What | State |
|---|---|---|
| 1 | Laptop pipeline: preprocessing, OCR, matcher, embedder, retriever, fusion, decision, `predict()`, evaluation | In progress (text branch and embedder done, rest pending) |
| 2 | FastAPI `/predict` endpoint plus Android client (two or more images in, JSON out) | Not started |
| 3 | On-device (ML Kit OCR, TFLite embedder, matcher port) | Future work only |

## Where the AI is

The project is a multimodal system. Each part is kept only if it beats a simpler alternative on our own measurements.

- **Computer vision preprocessing** (resize caps, upscale, optional contrast tools), compared experimentally.
- **OCR plus NLP-style text normalisation and fuzzy matching**, constrained to a known ingredient list.
- **Deep learning branch:** pretrained MobileNetV2 embeddings, cosine kNN retrieval, then a small Keras classifier head compared against kNN and SVM.
- **Fusion and decision layer** with abstention, thresholds calibrated on a validation set.
- **Evaluation:** ablation of text only vs image only vs fused, plus a failure analysis.

An honest note: the text branch is mostly classical (OCR, regex, fuzzy matching). The learned parts are the CNN branch, the trained head, and the evaluation. If the image branch adds little, we report that.

## Safety principles

1. Abstain over guess. A confident wrong answer is worse than "unrecognized".
2. Never present output as medical advice.
3. Do not invent facts about Nepali brands, manufacturers, or regulations. Unknown values are marked `TODO` and verified against a source.
4. Raw pack photos stay private (git-ignored). Strip location (EXIF) data before sharing images.
