# AlgoMatch: Agentic DSA Pattern Recommender

## Project Overview and Problem Statement
AlgoMatch is an AI-driven, Retrieval-Augmented Generation (RAG) platform designed to assist Software Engineers and students with identifying optimal Data Structures and Algorithms (DSA) for their specific software architecture problems. 

Often, developers understand the core constraints of a problem but struggle to identify the formal algorithmic pattern that optimally solves it. AlgoMatch bridges this gap by taking a natural language description of a software problem, conducting semantic similarity searches against a curated knowledge base of advanced computational patterns, and employing a local Agentic LLM (LLaMA 3) to reason about the best fit and generate rigorous, compiling C++ boilerplate code.

## System Design and Technical Approach
The architecture of AlgoMatch is completely localized and relies on three main components to ensure privacy and rapid inference:

1. **Frontend Interface (Streamlit):** A clean, dark-mode user interface that captures user queries, maintains chat history, and displays markdown-formatted AI responses.
2. **Vector Database Engine (Endee):** A high-performance, local vector database utilized to store and index our DSA knowledge base.
3. **Agentic LLM (Ollama & LLaMA 3):** Takes the retrieved structural context from Endee and executes a strict system prompt to evaluate tradeoffs and generate the final C++ code. The LLM acts as an expert software architect, adhering to complex rules (e.g., specific std::priority_queue implementations).
4. **Embedding Library (Sentence-Transformers):** Converts text into 384-dimensional dense vectors (`all-MiniLM-L6-v2`) for mathematical similarity comparison.

## Explanation of How Endee is Used
Endee serves as the core Retrieval engine for the RAG pipeline.

- **Initialization & Vector Generation:** Upon startup, `vector_engine.py` initializes the Endee client and creates a `dsa_patterns` index (dimension 384, cosine similarity, float32 precision). The predefined conceptual patterns inside `dsa_knowledge_base.py` are embedded using HuggingFace's `SentenceTransformer` and passed to Endee using the `index.upsert()` methodology alongside their rich metadata (complexities, definitions).
- **Retrieval:** When a user submits a query, it is immediately converted into a query vector and passed to Endee's `index.query()` API. Endee performs a lightning-fast cosine similarity search, returning the top matches. The metadata of these matches is injected into LLaMA 3's context window.

## Setup and Execution Instructions
This project requires Python 3.10+ and Docker (for the Endee server) to run locally.

### 1. Start the Endee Vector Database
Launch the official Endee server via Docker on port 8080:
```bash
docker run -p 8080:8080 -v endee-data:/data --name endee-server -d endeeio/endee-server:latest
```

### 2. Verify Ollama and LLaMA 3
Ensure Ollama is installed on your system. Pull the `llama3` model:
```bash
ollama pull llama3
```

### 3. Setup Python Environment
Navigate to the `AlgoMatch` directory, create a virtual environment, and install dependencies:
```bash
cd AlgoMatch
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit UI. The system will automatically build the Endee vector index on the first run.
```bash
streamlit run app.py
```
