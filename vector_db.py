# ================================================================
# 1) LOAD MEDLINEPLUS CSV (robust to missing/renamed columns)
# ================================================================
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')  # ← your cluster URL
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY') # ← your API key
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')

# client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# Connect to Qdrant Cloud (HTTPS + API key)
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def load_embedding_model():
    global _emb_model
    EMB_MODEL_NAME = "all-MiniLM-L6-v2"
    _emb_model = SentenceTransformer(EMB_MODEL_NAME)


# ----------------- Helper cleaner -----------------
def _clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).replace("\n", " ")
    return " ".join(s.split())

def process_data():
    DATA_PATH = Path("data") / "medlineplus_topics_2025-11-19.csv"

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"❌ Cannot find MedlinePlus CSV at {DATA_PATH}. "
            "Make sure it is inside the /data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print("Loaded MedlinePlus CSV with columns:", df.columns.tolist())



    # ----------------- Detect the REAL text column -----------------
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

    df["text"] = df[text_col].apply(_clean_text)

    
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
    return corpus

def embed_upsert_qdrant(corpus):
    
    # ================================================================
    # 3) EMBEDDINGS (loaded once when the module imports)
    # ================================================================
    
    _emb_matrix = _emb_model.encode(
        corpus["text"].tolist(),
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Generated embeddings:", _emb_matrix.shape)


    # ================================================================
    # 4) CREATE / UPSERT INTO QDRANT COLLECTION
    # ================================================================

    
    # Create or recreate the collection
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=_emb_matrix.shape[1],
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    print(f"Recreated Qdrant collection: {COLLECTION_NAME}")

    # Build Qdrant points
    points = []
    for i, row in corpus.iterrows():
        vector = _emb_matrix[i]
        payload = {
            "topic_id": row["topic_id"],
            "title": row["title"],
            "url": row["url"],
            "groups": row["groups"],
            "text": row["text"],
            "chunk_id": row["chunk_id"],
        }
        points.append(
            PointStruct(
                id=i,              # numeric IDs are ideal
                vector=vector,
                payload=payload,
            )
        )

    # Upsert into Qdrant
    # client.upsert(
    #     collection_name=COLLECTION_NAME,
    #     points=points,
    # )
    # Upsert in safe small batches
    BATCH = 100   # reduce to avoid 32 MB limit

    for start in range(0, len(points), BATCH):
        end = start + BATCH
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:end],
        )
        print(f"Upserted {min(end, len(points))}/{len(points)}")

    print(f"✅ Upserted {len(points)} vectors into Qdrant collection '{COLLECTION_NAME}'")

def semantic_search(query, top_k=6):
    hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query=_emb_model.encode(query).tolist(),
    limit=top_k,).points

    return hits

if __name__=="__main__":
    # corpus=process_data()
    # embed_upsert_qdrant(corpus)
    results=semantic_search("Fever and sore throat")
    print(results)