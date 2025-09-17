from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import List, TypedDict
class GraphState(TypedDict):
    '''
    Class chứa state của đồ thị
    '''
    
    documents: List[Document]
    question: str
    answer: str
    
class RAGChain:
    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever
        self.llm = llm
    
    def retrieve(self, state):
        
        print("Retrieving information")
        question = state["question"]
        
        if self.retriever is None:
            return {"documents": [], "question": question}       
        
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        assert self.llm is not None, "LLM is not set"
        print("Generating answer")
        
        question = state["question"]
        documents = state["documents"]
        
        # Create the prompt using the from_template class method
        # and use curly braces for variables.
        prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = prompt | self.llm | StrOutputParser()
        
        # Invoke the chain with the correct input variables
        answer = rag_chain.invoke({"context": documents, "question": question})
        
        # Return the result with the "answer" key to match GraphState
        return {"documents": documents, "question": question, "answer": answer}
    
    def build_graph(self):
        """Xây dựng đồ thị LangGraph."""
        workflow = StateGraph(GraphState)

        # Định nghĩa các nút (nodes)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        # Xây dựng các cạnh (edges)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Biên dịch đồ thị thành một ứng dụng có thể chạy được
        app = workflow.compile()
        return app


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    rag_chain = RAGChain(llm=llm)
    
    app = rag_chain.build_graph()
    initial_state = {"question": "Who are you?"}
    result = app.invoke(initial_state)
    print(result)