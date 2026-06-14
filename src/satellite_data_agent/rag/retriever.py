"""RAG retrieval chain using ChromaDB and LangChain LCEL."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


CHROMA_PERSIST_DIR = Path("data/chroma_db")
COLLECTION_NAME = "satellite_ops_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


def retrieve_context(query: str) -> str:
    """Retrieve relevant document chunks from ChromaDB for a given query.

    Returns concatenated text of top-K most similar chunks.
    If ChromaDB is not initialized or empty, returns empty string.
    """
    if not CHROMA_PERSIST_DIR.exists():
        return ""

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        return ""

    if collection.count() == 0:
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
    )

    documents = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(documents)
