from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.vectorstore.chroma_store import ChromaStore

class RAGPipeline:
    def __init__(self, vector_store: ChromaStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(temperature=0)
        
        self.prompt_template = """
        Answer the question based on the provided context. If you cannot find the answer in the context, say "I don't have enough information to answer this question."
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.collection.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt}
        )

    async def get_answer(self, question: str) -> str:
        return await self.qa_chain.arun(question) 