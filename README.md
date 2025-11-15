# Healthcare Symptom Checker Chatbot

A sophisticated healthcare symptom checker chatbot built with RAG (Retrieval-Augmented Generation) technology. This application uses medical textbooks as a knowledge base to provide educational information about symptoms, conditions, and medical management. The chatbot leverages Groq LLM, FAISS vector database, and Streamlit for an interactive user interface.

## 🎯 Project Overview

This project is an **educational medical information assistant** that helps users understand medical conditions, symptoms, and treatments based on authoritative medical textbooks. It uses advanced AI techniques to retrieve relevant information from medical literature and present it in an easy-to-understand format.


## ✨ Features

- **RAG-Powered Responses**: Uses Retrieval-Augmented Generation to provide accurate answers based on medical textbooks
- **Emergency Triage Detection**: Automatically detects emergency-related keywords and prompts users to seek immediate medical care
- **Structured Medical Information**: Provides organized responses including:
  - Expanded definitions
  - Symptoms
  - Possible conditions
  - Differential diagnoses
  - Stepwise management
  - Red flags
- **Interactive Web Interface**: User-friendly Streamlit-based chat interface
- **Fast Vector Search**: FAISS-based semantic search for quick information retrieval
- **Strict Retrieval-Only Mode**: Only provides information found in the uploaded medical books, preventing hallucinations

## 🏗️ Architecture

The project follows a three-stage architecture:

### 1. **Data Processing Stage** (`create_memory_for_llm.py`)
   - Loads PDF medical textbooks from the `data/` directory
   - Splits documents into chunks with overlap for better context preservation
   - Creates embeddings using HuggingFace sentence transformers
   - Builds and saves a FAISS vector database

### 2. **RAG Pipeline** (`connect_memory_with_llm.py`)
   - Loads the pre-built FAISS vectorstore
   - Retrieves relevant document chunks based on user queries
   - Uses Groq LLM to generate responses based on retrieved context
   - Implements emergency triage detection
   - Enforces strict retrieval-only responses

### 3. **User Interface** (`medibot.py`)
   - Streamlit-based web application
   - Chat interface for user interactions
   - Caches vectorstore for performance
   - Displays conversation history

## 📋 Prerequisites

- **Python**: 3.9 or higher (3.12 recommended)
- **Groq API Key**: Get your free API key from [Groq Console](https://console.groq.com/)
- **Medical PDFs**: Place medical textbook PDFs in the `data/` directory



## 🔧 Configuration

### Embedding Model

The default embedding model is `sentence-transformers/all-mpnet-base-v2`. To change it, modify the `EMBEDDING_MODEL_NAME` variable in:
- `create_memory_for_llm.py` (for building the vectorstore)
- `connect_memory_with_llm.py` (for querying)

### Chunking Parameters

Adjust chunk size and overlap in `create_memory_for_llm.py`:

```python
CHUNK_SIZE = 1200      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
```

### Retrieval Parameters

Modify the number of retrieved documents in `connect_memory_with_llm.py`:

```python
RETRIEVAL_K = 6  # Number of document chunks to retrieve
```

### LLM Configuration

Adjust Groq LLM settings in your `.env` file or `connect_memory_with_llm.py`:

- `GROQ_MODEL_NAME`: Model to use (default: `llama-3.1-8b-instant`)
- `GROQ_TEMPERATURE`: Response creativity (0.0-1.0, default: 0.3)
- `GROQ_MAX_TOKENS`: Maximum response length (default: 512)

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and RAG pipeline
- **Groq**: Fast LLM inference API
- **FAISS**: Vector similarity search (Facebook AI Similarity Search)
- **HuggingFace**: Embedding models and transformers
- **PyPDF**: PDF document processing
- **Python-dotenv**: Environment variable management

## 🔍 How It Works

1. **Document Processing**:
   - PDFs are loaded and split into manageable chunks
   - Each chunk is embedded into a high-dimensional vector space
   - Vectors are stored in FAISS for fast similarity search

2. **Query Processing**:
   - User query is embedded using the same model
   - FAISS searches for the most similar document chunks
   - Top K chunks are retrieved as context

3. **Response Generation**:
   - Retrieved chunks are passed to Groq LLM with a structured prompt
   - LLM generates a response based ONLY on the provided context
   - Response is formatted and displayed to the user

4. **Safety Features**:
   - Emergency keyword detection triggers immediate medical advice
   - Strict retrieval-only mode prevents hallucination
   - Medical disclaimers are automatically appended

