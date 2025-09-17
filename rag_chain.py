from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from typing import List, TypedDict

class GraphState(TypedDict):
    '''
    Class chứa state của đồ thị.
    Bổ sung 'conversation' để lưu lịch sử hội thoại.
    '''
    documents: List[Document]
    question: str
    answer: str
    conversation: List
    original_language: str

class RAGChain:
    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever
        self.llm = llm
        
    def translate_question(self, state):
        question = state["question"]
        # Nếu câu hỏi đã là tiếng Anh, không cần dịch
        if all('\u0000' <= char <= '\u007F' for char in question):
            state["original_language"] = "English"
            return state
        
        # Prompt để yêu cầu LLM dịch câu hỏi sang tiếng Anh
        translation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that translates questions into English."
                 " Only translate the question, do not answer it."),
                ("user", "{question}"),
            ]
        )
        
        language_identifier_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that identifies the language of the question."
                 " Only respond with the name of the language, do not answer the question."),
                ("user", "{question}"),
            ]
        )

        translator_chain = translation_prompt | self.llm | StrOutputParser()
        language_chain = language_identifier_prompt | self.llm | StrOutputParser()
        
        # Gọi chain để dịch câu hỏi
        translated_question = translator_chain.invoke({"question": question})
        original_language = language_chain.invoke({"question": question})

        # Cập nhật lại question trong state
        state["question"] = translated_question
        state["original_language"] = original_language
        
        print(f"Original question: {question}")
        print(f"Translated question: {translated_question}")
        print(f"Original language: {original_language}")
        
        return state

    def rewrite_question(self, state):
        question = state["question"]
        conversation = state["conversation"]

        if not conversation:
            return state

        # Prompt để yêu cầu LLM viết lại câu hỏi
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given a chat history and the latest user question "
                           "which might reference context in the chat history, "
                           "formulate a standalone question which can be understood "
                           "without the chat history. Do NOT answer the question, "
                           "just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder(variable_name="conversation"),
                ("user", "{question}"),
            ]
        )

        rewriter_chain = rewrite_prompt | self.llm | StrOutputParser()

        # Gọi chain để tạo câu hỏi mới
        rewritten_question = rewriter_chain.invoke({
            "conversation": [msg["content"] for msg in conversation],
            "question": question
        })
        
        print(f"Original question: {question}")
        print(f"Rewritten question: {rewritten_question}")

        # Cập nhật lại question trong state
        state["question"] = rewritten_question
        return state

    def retrieve(self, state):
        question = state["question"]
        
        if self.retriever is None:
            return {"documents": [], "question": question}       
        
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        assert self.llm is not None, "LLM is not set"
        
        question = state["question"]
        documents = state["documents"]
        conversation = state["conversation"]

        prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context and the chat history to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Chat History: {chat_history}
Question: {question}
Context: {context}
Answer:
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = prompt | self.llm | StrOutputParser()
        
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        answer = rag_chain.invoke({
            "context": documents, 
            "question": question,
            "chat_history": chat_history_str
        })
        
        retranslate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that translates answers into the original language."
                 " If the answer is in the original language, return the answer as is."
                 " Only translate the answer, do not answer the question."),
                ("user", "Original language:{original_language}"
                 " Answer: {answer}"),
            ]
        )
        
        retranslator_chain = retranslate_prompt | self.llm | StrOutputParser()
        if state.get("original_language") and state["original_language"].lower() != "english":
            answer = retranslator_chain.invoke({
                "answer": answer,
                "original_language": state["original_language"]
            })
        
        return {"documents": documents, "question": question, "answer": answer}

    def build_graph(self):
        """Xây dựng đồ thị LangGraph với bước rewrite_question."""
        workflow = StateGraph(GraphState)

        # Định nghĩa các nút (nodes)
        workflow.add_node("translate_question", self.translate_question)
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        # Xây dựng các cạnh (edges)
        
        workflow.set_entry_point("rewrite_question")
        workflow.add_edge("rewrite_question", "translate_question")
        workflow.add_edge("translate_question", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        app = workflow.compile()
        return app