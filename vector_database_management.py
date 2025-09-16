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
    return Chroma.from_documents(
        split_up_doc,
        embeddings_model,
        collection_name,
        db_path
    )

def load_vector_store(embeddings_model, collection_name, db_path):
    print("Loading vector store")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings_model,
        collection_name=collection_name
    )