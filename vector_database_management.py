from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import time

def load_document(markdown_path):
    return UnstructuredMarkdownLoader(markdown_path).load()

def split_document(document):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    ).split_documents(document)

def create_vector_store(split_up_doc, embeddings_model, collection_name, db_path):
    print("Creating vector store")
    
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings_model,
        collection_name=collection_name
    )
    
    # Xu ly rate limit cua GoogleStudio
    batch_size = 90
    total_texts = len(split_up_doc)
    
    for i in range(0, total_texts, batch_size):
        print(f"Processing batch {i  // batch_size}/ {total_texts // batch_size}")
        batch = split_up_doc[i:i+batch_size]
        vector_store.add_documents(batch)
        
        if i * batch_size < total_texts:
            time.sleep(60)
        
    return vector_store

def load_vector_store(embeddings_model, collection_name, db_path):
    print("Loading vector store")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings_model,
        collection_name=collection_name
    )