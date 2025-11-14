"""
connect_memory_with_llm.py — Production Version
Fully updated for LangChain v0.2+, no console prints, no interactive I/O.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ---------------- CONFIG ----------------

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RETRIEVAL_K = 6

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", 0.3))
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", 512))


# ---------------- EMERGENCY TRIAGE ----------------

RED_FLAG_KEYWORDS = [
    "chest pain", "shortness of breath", "loss of consciousness",
    "severe bleeding", "uncontrolled bleeding", "sudden weakness",
    "paralysis", "severe abdominal pain", "high fever",
    "difficulty breathing", "blue lips", "struggling to breathe",
]

def contains_red_flag(text: str):
    t = text.lower()
    return any(flag in t for flag in RED_FLAG_KEYWORDS)


# ---------------- PROMPT TEMPLATE ----------------

PROMPT_TEMPLATE = """
You are a careful medical AI assistant. Use ONLY the context provided.
If the info is missing, say you don't have enough information.

Format:
1) One-line definition
2) Symptoms (bulleted)
3) Top 3 differential diagnoses
4) Stepwise management
5) Red flags
6) Contraindications
7) Questions to ask a doctor
8) Sources

IMPORTANT:
- Do NOT hallucinate
- Do NOT add content outside context
- ALWAYS end with: "This information is educational only and does not replace professional medical advice."

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# ---------------- LOADERS ----------------

def load_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        api_key=GROQ_API_KEY,
    )


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return retriever


# ---------------- BUILD RAG CHAIN ----------------

def build_rag_chain(llm, retriever):
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


# ---------------- HELPERS ----------------

def extract_sources(docs):
    names = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        names.append(os.path.basename(src))
    return list(dict.fromkeys(names))


def confidence_from_docs(docs):
    n = len(docs)
    if n >= 6: return "High"
    if n >= 3: return "Medium"
    return "Low"


# ---------------- PUBLIC QUERY FUNCTION (USE THIS IN PRODUCTION) ----------------

def run_query(user_query: str):
    """
    Main function: returns structured medical answer + metadata
    """

    # --- Emergency check ---
    if contains_red_flag(user_query):
        return {
            "answer": "⚠ This query may indicate an emergency. Seek immediate medical care.",
            "sources": [],
            "confidence": "N/A",
        }

    llm = load_llm()
    retriever = load_vectorstore()

    docs = retriever.get_relevant_documents(user_query)
    if not docs:
        return {
            "answer": "No matching medical information was found in the uploaded books.",
            "sources": [],
            "confidence": "Low",
        }

    rag_chain = build_rag_chain(llm, retriever)
    response = rag_chain.invoke({"input": user_query})

    answer_text = response.get("answer") or response.get("output_text") or ""
    sources_list = extract_sources(docs)
    confidence = confidence_from_docs(docs)

    final_answer = (
        answer_text
        + f"\n\nSources: {', '.join(sources_list)}"
        + f"\nConfidence: {confidence}"
        + "\n\nNote: This information is educational only and does not replace professional medical advice."
    )

    return {
        "answer": final_answer,
        "sources": sources_list,
        "confidence": confidence,
    }
