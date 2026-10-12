"""Chroma vector store + retrieval (vector search, then cross-encoder reranking).

Embeddings and the reranker both run locally via sentence-transformers, so no
API key is needed. Models are loaded lazily and cached for the process lifetime.
"""
from functools import lru_cache
from pathlib import Path

import chromadb

DB_DIR = Path(__file__).parent / "chroma_db"   # persisted on disk, gitignored
COLLECTION = "ragas_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20   # how many vector hits the reranker gets to reorder


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    # cosine distance pairs with the normalized embeddings above
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def reset_collection():
    chromadb.PersistentClient(path=str(DB_DIR)).delete_collection(COLLECTION) if _exists() else None
    return collection()


def _exists() -> bool:
    return COLLECTION in [c.name for c in chromadb.PersistentClient(path=str(DB_DIR)).list_collections()]


def search(query: str, k: int = 4, rerank: bool = True) -> list[dict]:
    """Return the top-k chunks as dicts: text, source, heading, score.

    rerank=False gives plain vector search (used for the before/after comparison).
    """
    col = collection()
    if col.count() == 0:
        raise RuntimeError("The RAG index is empty. Run: python -m servers.rag_server.ingest")
    n = max(CANDIDATES, k) if rerank else k
    res = col.query(query_embeddings=embed([query]), n_results=min(n, col.count()))
    hits = [
        {"text": doc, "source": meta["source"], "heading": meta["heading"], "score": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    if rerank:
        # A cross-encoder reads (query, chunk) together, so it judges relevance far more
        # precisely than comparing two independent embeddings.
        scores = _reranker().predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]
