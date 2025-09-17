from dotenv import load_dotenv
from vector_database_management import load_document, split_document, load_vector_store, create_vector_store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_chain import RAGChain
import os
import config

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

if __name__ == "__main__":
    embedding_models = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"})

    vector_store = None

    # Kiểm tra xem thư mục db_path có tồn tại không, nếu không thì tạo mới csdl
    if not os.path.exists(config.VECTOR_DB_PATH):
        os.makedirs(config.VECTOR_DB_PATH)

        # Load document từ file markdown
        document = load_document(config.D)

        # Chia nhỏ document
        split_up_doc = split_document(document)

        # Tạo vector store và lưu trữ
        vector_store = create_vector_store(
            split_up_doc, 
            embedding_models, 
            config.VECTOR_DB_COLLECTION, 
            config.VECTOR_DB_PATH)

    else:
        # Nếu thư mục db_path đã tồn tại, tải vector store từ thư mục
        vector_store = load_vector_store(
            embedding_models, 
            config.VECTOR_DB_COLLECTION, 
            config.VECTOR_DB_PATH)

    # Kiểm tra vector_store đã được khởi tạo chưa
    assert vector_store is not None, "Vector store should be initialized"
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE)
    
    rag_chain = RAGChain(llm=llm, retriever=retriever)
    app = rag_chain.build_graph()
    
    # Chạy ứng dụng
    question = "How can I use n8n to automate tasks?"
    print(f"Câu hỏi: {question}")
    
    inputs = {"question": question}
    for output in app.stream(inputs, {"recursion_limit": 5}):
        for key, value in output.items():
            print(f"Kết quả từ nút: {key} là: {value}")

    final_state = value
    print("--------------------------------------------------")
    print(f"Câu trả lời cuối cùng: {final_state.get('answer', 'Không có câu trả lời được tạo ra.')}")
    print("--------------------------------------------------")
    
