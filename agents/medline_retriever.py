import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================================================================
# 1) LOAD MEDLINEPLUS CSV (robust to missing/renamed columns)
# ================================================================
DATA_PATH = Path("data") / "medlineplus_topics_2025-11-19.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"❌ Cannot find MedlinePlus CSV at {DATA_PATH}. "
        "Make sure it is inside the /data folder."
    )

df = pd.read_csv(DATA_PATH)

print("Loaded MedlinePlus CSV with columns:", df.columns.tolist())

# ----------------- Helper cleaner -----------------
def _clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).replace("\n", " ")
    return " ".join(s.split())


# ----------------- Detect the REAL text column -----------------
# Try a list of candidates based on MedlinePlus structure variations
TEXT_CANDIDATES = [
    "text",
    "full-summary",
    "full_summary",
    "summary",
    "body",
    "content",
    "article",
]

text_col = None
for cand in TEXT_CANDIDATES:
    if cand in df.columns:
        text_col = cand
        break

if text_col is None:
    raise ValueError(
        "❌ Could not find any valid text column. "
        "Looked for: " + str(TEXT_CANDIDATES) +
        f"\nAvailable columns: {df.columns.tolist()}"
    )

# Standardize → Always make df["text"] exist
df["text"] = df[text_col].apply(_clean_text)


# ----------------- Optional columns -----------------
if "also_called" in df.columns:
    df["also_called"] = df["also_called"].apply(_clean_text)
else:
    df["also_called"] = ""

if "groups" not in df.columns:
    df["groups"] = ""


# ================================================================
# 2) CHUNK THE TEXT
# ================================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)

records = []
for _, row in df.iterrows():
    chunks = splitter.split_text(row["text"] or "")
    for j, ch in enumerate(chunks):
        records.append({
            "topic_id": row.get("id", ""),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "groups": row.get("groups", ""),
            "chunk_id": f"{row.get('id', 'UNK')}_{j}",
            "text": ch,
        })

corpus = pd.DataFrame(records)


# ================================================================
# 3) EMBEDDINGS (loaded once when the module imports)
# ================================================================
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_emb_model = SentenceTransformer(EMB_MODEL_NAME)

_emb_matrix = _emb_model.encode(
    corpus["text"].tolist(),
    show_progress_bar=False,
    normalize_embeddings=True,
).astype(np.float32)


# ================================================================
# 4) RETRIEVAL FUNCTION (the only function planner calls)
# ================================================================
def retrieve_passages(query: str, top_k: int = 6) -> List[Dict]:
    """
    Retrieve top_k relevant MedlinePlus chunks.
    Returns [{title, url, text, score}, ...]
    """
    if not query:
        return []

    q_vec = _emb_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, _emb_matrix)[0]
    top_idx = np.argsort(-sims)[:top_k]

    subset = corpus.iloc[top_idx].copy()
    subset["score"] = sims[top_idx]

    results = []
    for _, row in subset.iterrows():
        results.append({
            "title": row["title"],
            "url": row["url"],
            "text": row["text"],
            "score": float(row["score"]),
        })

    return results