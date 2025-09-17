import os
from dotenv import load_dotenv

load_dotenv()

TEMPERATURE = 0.3
LLM_MODEL="gemini-2.5-flash"
EMBEDDINGS_MODEL="sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_PATH="Document/n8n_docs_combined.md"
VECTOR_DB_PATH="Database"
VECTOR_DB_COLLECTION="collection"