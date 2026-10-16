import re 

import pandas as pd  
from rapidfuzz import fuzz 

OCR_FIXES = str.maketrans({"0": "o", "1": "l", "5": "s"})
STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(mcg|mg|ml|g|%)", re.IGNORECASE)

def normalize(text: str):
    """Lowercase and collapse whitespace"""
    return re.sub(r"\s+", " ", text.lower()).strip() 


def extract_strengths(text: str):
    """Return strengths like ["500 mg",  "0.05 %"] found in raw text."""
    return [f"{num} {unit.lower()}" for num, unit in STRENGTH_RE.findall(text)]


def _tokens(text: str):
    """Lowercase letter only tokens with OCR digit confusion repaired"""
    tokens = [] 
    for raw in normalize(text).split(" "):
        if re.search(r"[a-z]", raw):
            letters = re.sub(r"[^a-z]", "", raw.translate(OCR_FIXES))
            if letters:
                tokens.append(letters)

    return tokens


def _ngrams(tokens: list[str], max_n: int):
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i:i + n])


def _score(gram: str, name: str):
    """Whole string similarity (0...1). Truncated prefix socre 0.8-0.9 and exact reads 1.0"""
    g = gram.replace(" ", "")
    n = name.replace(" ", "")
    best = fuzz.ratio(g, n) / 100 
    if 5 <= len(g) < len(n):
        coverage = len(g) / len(n)
        if coverage >= 0.5 and fuzz.ratio(g, n[:len(g)]) >= 90:
            best = max(best, 0.7 + 0.2 * coverage)

    return best 


def text_scores(text: str, ref_db: dict[str, list[str]], min_ratio: int = 80, min_len: int = 4, max_ngram: int = 3):
    """
    Best whole-string match per ingredient
    Score below min_ratio / 100 are to be dropped
    """
    tokens = _tokens(text)
    scores: dict[str, float] = {} 
    for ingredient, names in ref_db.items():
        best = 0.0 
        for gram in _ngrams(tokens, max_ngram):
            if len(gram) < min_len:
                continue 
            for name in names:
                s = _score(gram, name) 
                if s * 100 >= min_ratio: 
                    best = max(best, s)
        if best > 0:
            scores[ingredient] = best  

    return scores 


def load_name_index(csv_path):
    """Build {ingredient : [lowwercase names including aliases]} from reference_db.csv"""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    index: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        names = {row["ingredient"]} | {a for a in row["aliases"].split("|") if a}
        index.setdefault(row["ingredient"], set()).update(n.strip().lower() for n in names)
    
    return {k: sorted(v) for k, v in index.items()}     