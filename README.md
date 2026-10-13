# MediScan

## Requirements 
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Tesseract OCR (system program)

## Setup 

```bash
uv venv --python 3.12
uv pip install -r requirements.txt 
```

Activate with `source .venv/bin/activate` (Linux) or `.venv\Scripts\activate` (Windows).

Install Tesseract:

- Linux: install tesseract-ocr using package manager of distro
- Windows: UB Mannheim installer, then add it to PATH 

## Verify

```bash
python -c "import src; print('ok')"
pytest
```

## Notes

- Name your environment `.venv`
- Add new packages to `requirements.txt` in the same commit 
- `data/raw/` is not in git. Shared via drive link
