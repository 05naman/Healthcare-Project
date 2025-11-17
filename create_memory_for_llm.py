"""
create_memory_for_llm.py
- Loads PDFs from /data
- Splits into optimized medical chunks
- Creates embeddings using HuggingFace
- Builds FAISS vectorstore for retrieval
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- CONFIG ----------------

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Tuned for medical textbooks
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ⭐ Change this model to BioBERT if you want stronger medical accuracy
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ---------------- LOAD PDF FILES ----------------

def load_pdf_files(data_path: str):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        recursive=False,
    )
    documents = loader.load()
    print(f"[load_pdf_files] Loaded {len(documents)} pages/documents from {data_path}")
    return documents


# ---------------- CREATE TEXT CHUNKS ----------------

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add metadata (source filename, chunk index)
    for i, doc in enumerate(chunks):
        md = dict(doc.metadata) if doc.metadata else {}

        # Normalize metadata
        if "source" not in md:
            md["source"] = md.get("file_path", "unknown_source")

        md["chunk_index"] = i
        doc.metadata = md

    print(f"[create_chunks] Produced {len(chunks)} text chunks")
    return chunks


# ---------------- BUILD FAISS VECTORSTORE ----------------

def build_faiss_vectorstore(chunks):
    print(f"[build_faiss_vectorstore] Embedding model: {EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print(f"[build_faiss_vectorstore] Saved FAISS db to {DB_FAISS_PATH}")
    return db


# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("=== Building FAISS Vectorstore from medical PDFs ===")

    docs = load_pdf_files(DATA_PATH)
    chunks = create_chunks(docs)
    db = build_faiss_vectorstore(chunks)

    print("Done. You can now run connect_memory_with_llm.py")
