"""
create_memory_for_llm.py
- Loads PDFs from data/
- Splits into larger chunks with overlap
- Attaches simple metadata (source file, page)
- Builds embeddings and saves FAISS vectorstore
"""

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

DATA_PATH = "data/"            # folder where you put PDFs
DB_FAISS_PATH = "vectorstore/db_faiss"

# Chunking params (tuned for medical books like Harrison's)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Choose a stronger embedding model.
# all-mpnet-base-v2 is a good general-purpose step up from MiniLM.
# If you find a medical embedding (Bio/Clinical) on HF, substitute here.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_pdf_files(data_path: str):
    """
    Load all PDF pages as LangChain Document objects from a directory.
    """
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()  # returns list of Documents, each usually a page
    print(f"[load_pdf_files] Loaded {len(documents)} pages/documents from {data_path}")
    return documents


def create_chunks(documents):
    """
    Split documents into larger chunks preserving overlap.
    Also attach metadata: source filename and original page (if available).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # split_documents will preserve metadata from original Documents
    chunks = text_splitter.split_documents(documents)

    # Ensure metadata contains filename/page information (best-effort)
    for i, doc in enumerate(chunks):
        # if origin metadata has 'source' or 'file_path' keep it, else mark unknown
        md = dict(doc.metadata) if doc.metadata else {}
        # try common keys from PyPDFLoader
        if "source" not in md and "file_path" in md:
            md["source"] = md.get("file_path")
        if "source" not in md:
            md["source"] = md.get("source", "unknown_source")
        # add chunk index for traceability
        md.setdefault("chunk_index", i)
        doc.metadata = md

    print(f"[create_chunks] Produced {len(chunks)} text chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_faiss_vectorstore(chunks, embedding_model_name=EMBEDDING_MODEL_NAME, save_path=DB_FAISS_PATH):
    """
    Create embeddings and store in a local FAISS vectorstore.
    """
    print(f"[build_faiss_vectorstore] Using embedding model: {embedding_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)
    print(f"[build_faiss_vectorstore] Saved FAISS vectorstore to {save_path}")
    return db


if __name__ == "__main__":
    # 1. load pdfs
    docs = load_pdf_files(DATA_PATH)

    # 2. create chunks
    text_chunks = create_chunks(docs)

    # 3. build & save FAISS vectorstore
    db = build_faiss_vectorstore(text_chunks)

    print("Done. You can now run connect_memory_with_llm.py to query the vectorstore.")
