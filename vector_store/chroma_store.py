"""ChromaDB vector store for semantic search over news articles (RAG)."""

from __future__ import annotations

import hashlib
import logging

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBEDDING_MODEL, DB_PATH

logger = logging.getLogger(__name__)

_collection = None
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def index_news(news_df: pd.DataFrame, batch_size: int = 50) -> int:
    """Embed and store news articles in ChromaDB for retrieval."""
    encoder = _get_encoder()
    collection = _get_collection()

    indexed = 0
    for i in range(0, len(news_df), batch_size):
        batch = news_df.iloc[i:i + batch_size]
        texts = (batch["title"].fillna("") + ". " + batch["content"].fillna("")).tolist()
        ids = [_text_hash(t) for t in texts]
        metadatas = [
            {
                "ticker": row.get("ticker", ""),
                "source": row.get("source", ""),
                "date": str(row.get("published_at", "")),
                "title": row.get("title", "")[:200] if isinstance(row.get("title"), str) else "",
            }
            for _, row in batch.iterrows()
        ]
        embeddings = encoder.encode(texts).tolist()

        collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=texts, metadatas=metadatas,
        )
        indexed += len(batch)

    logger.info("Indexed %d news articles in vector store", indexed)
    return indexed


def search(
    query: str,
    top_k: int = 5,
    ticker_filter: str | None = None,
) -> list[dict]:
    """Semantic search over news articles. Returns relevant articles with scores."""
    encoder = _get_encoder()
    collection = _get_collection()

    query_embedding = encoder.encode([query]).tolist()[0]
    where = {"ticker": ticker_filter} if ticker_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    articles = []
    if results and results.get("ids"):
        for i, doc_id in enumerate(results["ids"][0]):
            articles.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "relevance_score": 1.0 - results["distances"][0][i],
            })
    return articles


def reset_store():
    """Clear all data from the vector store."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("news_articles")
    except Exception:
        pass
    global _collection
    _collection = None
