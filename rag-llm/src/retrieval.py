import faiss
import pickle
import os
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# -------- PATH SETUP --------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

# -------- LOAD VECTOR STORE --------

index = faiss.read_index(os.path.join(VECTOR_STORE_DIR, "index.faiss"))

with open(os.path.join(VECTOR_STORE_DIR, "metadata.pkl"), "rb") as f:
    METADATA = pickle.load(f)

with open(os.path.join(VECTOR_STORE_DIR, "texts.pkl"), "rb") as f:
    TEXTS = pickle.load(f)

# -------- LOAD EMBEDDING MODEL --------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.path.join(BASE_DIR, "hf_cache"),
    device="cpu"
)

# -------- QUERY EMBEDDING CACHE --------

@lru_cache(maxsize=1000)
def get_query_embedding(query: str):
    """Cache query embeddings to avoid recomputing for repeated queries."""
    return model.encode([query])

# -------- RETRIEVER FUNCTION --------

def retrieve(
    query: str,
    top_k: int = 7,  # Reduced from 10 to 7 for faster performance
    filters: dict | None = None
):
    """
    filters example:
    {
        "knowledge_type": "company_facts",
        "batch_year": 2024,
        "company": "TCS"
    }
    """

    query_embedding = get_query_embedding(query)
    
    # When filtering, search more results to ensure we get enough matches
    # after filtering (since many results might be filtered out)
    # Increased to 15x because some knowledge types (like statistics) have very few chunks
    search_k = top_k * 15 if filters else top_k  # Search 15x more when filtering
    
    distances, indices = index.search(query_embedding, search_k)

    results = []

    for idx in indices[0]:
        if idx == -1:
            continue

        metadata = METADATA[idx]

        # Apply metadata filters
        if filters:
            skip = False
            for key, value in filters.items():
                if metadata.get(key) != value:
                    skip = True
                    break
            if skip:
                continue

        results.append({
            "text": TEXTS[idx],
            "metadata": metadata
        })
        
        # Stop once we have enough results
        if len(results) >= top_k:
            break

    return results


