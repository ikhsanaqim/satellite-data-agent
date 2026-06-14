"""Document ingestion and ChromaDB indexing for the RAG pipeline.

Usage:
    python -m satellite_data_agent.rag.indexer

This reads all .md and .txt files from data/knowledge_base/,
splits them into chunks, embeds them with all-MiniLM-L6-v2,
and stores them in a persistent ChromaDB collection.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Config
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
CHROMA_PERSIST_DIR = Path("data/chroma_db")
COLLECTION_NAME = "satellite_ops_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_index() -> None:
    """Read documents, chunk, embed, and store in ChromaDB."""
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        return

    # Load documents
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    if not docs:
        print("No documents found in knowledge base.")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # Create ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Delete collection if exists (re-index from scratch)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Add chunks to collection
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[{"source": chunk.metadata.get("source", "unknown")} for chunk in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    build_index()
