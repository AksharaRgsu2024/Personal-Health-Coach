import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------- 1. Load MedlinePlus topics from CSV ---------
# Adjust if your CSV lives somewhere else
DATA_PATH = Path("data") / "medlineplus_topics.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cannot find MedlinePlus CSV at {DATA_PATH}. "
                            "Make sure it is in the 'data' folder.")

df = pd.read_csv(DATA_PATH)

def _clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).replace("\n", " ")
    return " ".join(s.split())

df["text"] = df["text"].apply(_clean_text)
df["also_called"] = df["also_called"].apply(_clean_text)

# --------- 2. Chunk the text for retrieval ---------
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
            "topic_id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "groups": row["groups"],
            "chunk_id": f"{row['id']}_{j}",
            "text": ch,
        })

corpus = pd.DataFrame(records)

# --------- 3. Build embeddings once at import time ---------
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_emb_model = SentenceTransformer(EMB_MODEL_NAME)

_emb_matrix = _emb_model.encode(
    corpus["text"].tolist(),
    show_progress_bar=False,
    normalize_embeddings=True,
).astype(np.float32)


def retrieve_passages(query: str, top_k: int = 6) -> List[Dict]:
    """
    Retrieve top_k MedlinePlus chunks for a query.
    Returns a list of dicts: {title, url, text, score}.
    """
    if not query:
        return []

    q_vec = _emb_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, _emb_matrix)[0]
    top_idx = np.argsort(-sims)[:top_k]
    subset = corpus.iloc[top_idx].copy()
    subset["score"] = sims[top_idx]

    results: List[Dict] = []
    for _, row in subset.iterrows():
        results.append(
            {
                "title": row["title"],
                "url": row["url"],
                "text": row["text"],
                "score": float(row["score"]),
            }
        )
    return results
