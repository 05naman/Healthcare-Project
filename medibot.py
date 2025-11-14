import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from connect_memory_with_llm import run_query   # <-- use your production RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    # must match the embedding used during FAISS creation
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def main():
    st.title("Healthcare Symptom Checker Chatbot with Groq LLM and FAISS")
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # show past messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Ensure vectorstore loads (this warms cache)
            _ = get_vectorstore()

            # use your production RAG pipeline
            answer = run_query(prompt)

            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
