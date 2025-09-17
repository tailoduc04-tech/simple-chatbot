from dotenv import load_dotenv
from vector_database_management import load_document, split_document, load_vector_store, create_vector_store
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import config

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

if __name__ == "__main__":
     embedding_models = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

     vector_store = None

     # Kiểm tra xem thư mục db_path có tồn tại không, nếu không thì tạo mới csdl
     if not os.path.exists(config.DB_PATH):
        os.makedirs(config.DB_PATH)

        # Load document từ file markdown
        document = load_document(config.MARKDOWN_PATH)

        # Chia nhỏ document
        split_up_doc = split_document(document)

        # Tạo vector store và lưu trữ
        vector_store = create_vector_store(
            split_up_doc, 
            embedding_models, 
            config.COLLECTION_NAME, 
            config.DB_PATH)

     else:
        # Nếu thư mục db_path đã tồn tại, tải vector store từ thư mục
        vector_store = load_vector_store(
            embedding_models, 
            config.COLLECTION_NAME, 
            config.DB_PATH)

     # Kiểm tra vector_store đã được khởi tạo chưa
     assert vector_store is not None, "Vector store should be initialized"
     
     
        
