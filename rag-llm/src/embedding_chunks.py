import json
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=HF_CACHE_DIR,
    device="cpu"
)


CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_all_chunks():
    all_texts = []
    all_metadata = []

    for file in os.listdir(CHUNKS_DIR):
        if file.endswith(".json"):
            path = os.path.join(CHUNKS_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    all_texts.append(item["text"])
                    all_metadata.append(item["metadata"])

    return all_texts, all_metadata

def main():
    print("[INFO] Loading chunks...")
    texts, metadata = load_all_chunks()

    print(f"[INFO] Total chunks loaded: {len(texts)}")

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]

    print("[INFO] Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))

    with open(os.path.join(VECTOR_STORE_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    with open(os.path.join(VECTOR_STORE_DIR, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    print("[OK] Vector store created successfully")

if __name__ == "__main__":
    main()
