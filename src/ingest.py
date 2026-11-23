"""Knowledge base ingestion and local vector database initialization.

Loads fintech policy documentation, applies semantic text splitting,
computes local sentence-transformers embeddings (zero API keys, 100% offline),
and stores vectors into a persistent ChromaDB store.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_policy.md"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-loaded embedding instance to prevent duplicate model loads in memory
_CACHED_EMBEDDINGS = None


def get_embedding_function():
    """Initializes and caches the local HuggingFace embedding model.

    Runs 100% locally on CPU without requiring external API keys.
    """
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is not None:
        return _CACHED_EMBEDDINGS

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. Please ensure 'langchain-huggingface' "
                "or 'langchain-community' and 'sentence-transformers' are installed."
            ) from exc

    print(f"📦 [Embeddings] Loading local model: {EMBEDDING_MODEL_NAME}...")
    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _CACHED_EMBEDDINGS


def get_chroma_class():
    """Dynamically resolves Chroma vector store class across langchain versions."""
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma
        except ImportError as exc:
            raise ImportError(
                "ChromaDB wrapper not found. Please install 'langchain-chroma' or 'chromadb'."
            ) from exc


def get_text_splitter():
    """Returns a configured RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=70,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        keep_separator=True,
    )


def ingest_knowledge_base(
    data_path: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    force_reload: bool = False,
):
    """Loads markdown policy document, chunks text, and stores in ChromaDB.

    Args:
        data_path: Path to markdown file. Defaults to data/sample_policy.md.
        persist_dir: Path to Chroma directory. Defaults to ./chroma_db.
        force_reload: If True, resets existing collection and rebuilds.

    Returns:
        Chroma vector store instance.
    """
    data_file = Path(data_path) if data_path else DEFAULT_DATA_PATH
    db_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_DIR

    if not data_file.exists():
        raise FileNotFoundError(f"Knowledge document not found at: {data_file}")

    print(f"📄 [Ingest] Reading policy file: {data_file}")
    content = data_file.read_text(encoding="utf-8")

    splitter = get_text_splitter()
    raw_chunks = splitter.split_text(content)

    print(f"✂️  [Ingest] Generated {len(raw_chunks)} semantic chunks.")

    # Attach contextual metadata
    metadatas = []
    documents = []
    for idx, chunk in enumerate(raw_chunks):
        cleaned = chunk.strip()
        if not cleaned:
            continue
        documents.append(cleaned)
        metadatas.append({
            "source": data_file.name,
            "chunk_id": idx,
            "char_count": len(cleaned),
        })

    embeddings = get_embedding_function()
    ChromaClass = get_chroma_class()

    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 [Ingest] Saving embeddings into ChromaDB at: {db_dir}")
    vector_store = ChromaClass.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(db_dir),
        collection_name="fintech_policy_collection",
    )

    print(f"✅ [Ingest] Ingestion complete! {len(documents)} chunks indexed successfully.")
    return vector_store


def ensure_vector_store(
    persist_dir: Optional[Path] = None,
    data_path: Optional[Path] = None,
):
    """Auto-ingestion helper: checks if ChromaDB is populated; if not, ingests automatically.

    Guarantees that downstream services (FastAPI & Streamlit) never crash due to
    an uninitialized vector store.
    """
    db_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_DIR
    data_file = Path(data_path) if data_path else DEFAULT_DATA_PATH

    ChromaClass = get_chroma_class()
    embeddings = get_embedding_function()

    # If directory doesn't exist or is empty, run ingestion
    needs_ingestion = False
    if not db_dir.exists():
        needs_ingestion = True
    else:
        # Check if directory contains chroma database files
        items = list(db_dir.glob("*"))
        if len(items) == 0:
            needs_ingestion = True

    if needs_ingestion:
        print(f"⚡ [Auto-Ingest] Vector store not found. Running initial ingestion...")
        return ingest_knowledge_base(data_file, db_dir)

    # If folder exists, attempt to connect and check count
    try:
        vector_store = ChromaClass(
            persist_directory=str(db_dir),
            embedding_function=embeddings,
            collection_name="fintech_policy_collection",
        )
        count = vector_store._collection.count()
        if count == 0:
            print(f"⚡ [Auto-Ingest] Vector store empty ({count} items). Re-ingesting...")
            return ingest_knowledge_base(data_file, db_dir)
        return vector_store
    except Exception as exc:
        print(f"⚠️  [Auto-Ingest] Failed loading existing store ({exc}). Re-creating...")
        return ingest_knowledge_base(data_file, db_dir, force_reload=True)


if __name__ == "__main__":
    print("🚀 [Ingest Runner] Executing standalone vector ingestion...")
    store = ingest_knowledge_base(force_reload=True)
    query = "What is the fee for express same-day settlement?"
    results = store.similarity_search(query, k=2)
    print("\n🔍 Test Query Results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)
