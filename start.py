from dotenv import load_dotenv
from vector_database_management import load_document, split_document, load_vector_store, create_vector_store
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import config

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

if __name__ == "__main__":
    document = load_document(config.DOCUMENT_PATH)
    print(f"Loaded document with length {len(document[0].page_content)}")
    split_up_document = split_document(document)
    print(f"Split up document into {len(split_up_document)} seperate documents")
    embeddings_model = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL, model_kwargs={'device': 'cpu'})
    vector_store = create_vector_store(
            split_up_document, 
            embeddings_model,
            config.VECTOR_DB_COLLECTION, 
            config.VECTOR_DB_PATH
            ) if not os.path.exists(
                    os.path.join(config.VECTOR_DB_PATH, config.VECTOR_DB_COLLECTION)
                                ) else load_vector_store(
                                        embeddings_model,
                                        config.VECTOR_DB_COLLECTION, 
                                        config.VECTOR_DB_PATH)
