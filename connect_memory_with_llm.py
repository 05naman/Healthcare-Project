"""
connect_memory_with_llm.py — Production Version (ONLY FINAL ANSWER)
No sources. No confidence. Clean output for Streamlit.
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
You are a STRICT retrieval-only medical assistant.

RULES (IMPORTANT):
- You MUST use ONLY the information found in the provided context.
- If the user asks something that is NOT in the context, reply:
  "The provided medical books do not contain this information."
- You MUST NOT add external medical knowledge, treatments, pathogens, or assumptions.
- NO guessing. NO hallucination.
- NO home remedies, no homeopathy, no herbs, no oxygen therapy unless explicitly in the context.
- If symptoms do not match any conditions in context, say so.

-----------------------------------------------

Your output MUST follow this structure:

1) **Expanded Definition**
   - Rewrite the definition using MORE descriptive, simple language.
   - Use only the meaning found in the context.
   - You may expand or clarify the ideas, but you must NOT add facts not supported by the text.

2) **Symptoms (ONLY from context)**

3) **Possible Conditions Based on User Symptoms (ONLY if symptoms appear in context)**  
   - Match symptoms to conditions from the books only.
   - If symptoms do not appear in context, say:
     "The books do not contain enough information to match these symptoms."

4) **Top 3 Differential Diagnoses (ONLY from context)**

5) **Stepwise Management (ONLY if described in context)**  
   - If management is not discussed in context:
     “The provided books do not describe management steps for this condition.”

6) **Red Flags (ONLY if present in the context)**

-----------------------------------------------

<context>
{context}
</context>

User Question: {input}

-----------------------------------------------

REMEMBER:
If ANY part of the answer is not present in the context, you MUST explicitly say:
"The provided medical books do not contain this information."

Always end with:
"This information is educational only and does not replace professional medical advice."
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


# ---------------- MAIN PRODUCTION FUNCTION (ONLY FINAL ANSWER) ----------------

def run_query(user_query: str):
    """
    Main function: returns the *formatted medical answer* only.
    No confidence. No sources. Only final answer.
    """

    # Emergency triage
    if contains_red_flag(user_query):
        return "⚠ This query may indicate an emergency. Seek immediate medical care."

    # Load LLM + vectorstore
    llm = load_llm()
    retriever = load_vectorstore()

    docs = retriever.get_relevant_documents(user_query)
    if not docs:
        return "No matching medical information was found in the uploaded books."

    # Run the RAG pipeline
    rag_chain = build_rag_chain(llm, retriever)
    response = rag_chain.invoke({"input": user_query})

    answer_text = response.get("answer") or response.get("output_text") or ""

    # Ensure disclaimer is added
    final_answer = (
        answer_text +
        "\n\nNote: This information is educational only and does not replace professional medical advice."
    )

    return final_answer
