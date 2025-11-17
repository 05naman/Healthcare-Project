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
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", 1024))


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
You are a STRICT retrieval-only medical assistant that provides clear, user-friendly explanations.

RULES (CRITICAL):
- You MUST use ONLY information found in the provided context.
- Write in simple, clear language that ordinary people can understand.
- If information is NOT in the context, omit that section entirely (do not say "not found").
- You MUST NOT add external medical knowledge, treatments, or assumptions.
- NO guessing. NO hallucination. NO home remedies unless explicitly in context.

-----------------------------------------------

ANALYZE THE USER'S QUESTION TYPE:
- Definition/What is X? → Focus on definition, symptoms, causes (if in context)
- Symptom-based query → Focus on matching conditions, differential diagnoses
- Treatment/Management query → Focus on management, medications, procedures
- General query → Provide comprehensive information from context

-----------------------------------------------

Your output structure (ADAPT based on query type - only include relevant sections):

**1. What is [Condition]?**
   - Explain in simple, everyday language using information from the context.
   - Make it clear and easy to understand. Avoid overly technical jargon.
   - If definition not in context, skip this section.

**2. Common Symptoms**
   - List symptoms clearly in bullet points or numbered format.
   - Only include symptoms mentioned in the context.
   - If no symptoms in context, skip this section.

**3. Possible Causes or Types** (if mentioned in context)
   - Only include if this information exists in the context.
   - If not present, skip entirely.

**4. Related Conditions or Differential Diagnoses** (only if relevant to query)
   - Include this ONLY if:
     a) User asked about symptoms and matching conditions exist in context, OR
     b) The context discusses differential diagnoses for the condition.
   - If not relevant or not in context, skip this section.

**5. Treatment and Management**
   - List treatment options, medications, or management steps from context.
   - Present in a clear, organized way (numbered or bulleted).
   - If not in context, skip this section.

**6. When to Seek Immediate Care (Red Flags)**
   - Include this ONLY if the context mentions warning signs or emergency situations.
   - If not in context, skip this section.

-----------------------------------------------

<context>
{context}
</context>

User Question: {input}

-----------------------------------------------

IMPORTANT INSTRUCTIONS:
- Extract ALL relevant information from the context for each section.
- Only create sections where you have actual information from the context.
- Write naturally and conversationally - make it easy for people to understand.
- Do NOT include sections with "no information" messages - just skip those sections.
- Format your answer clearly with proper spacing and structure.
- Always end with a disclaimer about seeking professional medical advice.
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

    docs = retriever.invoke(user_query)


    if not docs:
        return "No matching medical information was found in the uploaded books."

    # Run the RAG pipeline
    rag_chain = build_rag_chain(llm, retriever)
    response = rag_chain.invoke({"input": user_query})

    answer_text = response.get("answer") or response.get("output_text") or ""

    # Ensure disclaimer is added (check if already present to avoid duplication)
    if "educational only" not in answer_text.lower() and "professional medical advice" not in answer_text.lower():
        final_answer = (
            answer_text +
            "\n\n⚠️ **Important:** This information is for educational purposes only and does not replace professional medical advice. Always consult a healthcare provider for proper diagnosis and treatment."
        )
    else:
        final_answer = answer_text

    return final_answer
