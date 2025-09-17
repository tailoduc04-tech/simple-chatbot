# Simple Chatbot Project

## Project Structure

- `start.py` — Main entry point for running the chatbot.
- `rag_chain.py` — Implements the RAG chain logic for retrieval and generation.
- `vector_database_management.py` — Handles vector database operations (embedding, storage, search).
- `config.py` — Configuration settings for the project.
- `Database/` — Contains the Chroma vector database files.
- `Document/` — Contains source documents for the knowledge base.

## Getting Started

1. **Install dependencies**:
	```bash
	pip install -r requirements.txt
	```

2. **Run the chatbot**:
	```bash
	python start.py
	```

## TODOs

- [ ] Deploy webhook endpoint