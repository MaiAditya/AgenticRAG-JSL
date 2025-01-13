from typing import Dict, Any
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.retrieval.hybrid_retriever import HybridRetriever

class AdvancedRAGPipeline:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(temperature=0)
        
        self.cot_prompt = PromptTemplate(
            template="""Let's approach this step by step:
            1. First, analyze the question carefully
            2. Then, examine the relevant context
            3. Think about the relationships between different pieces of information
            4. Form a comprehensive answer

            Question: {question}
            
            Context: {context}
            
            Let's think through this:
            1) What are the key points in the question?
            2) What relevant information do we have in the context?
            3) How do these pieces fit together?
            4) What is the most accurate and complete answer?

            Final Answer:""",
            input_variables=["question", "context"]
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.cot_prompt)

    def _extract_reasoning(self, response: str) -> list:
        # Extract reasoning steps from response
        steps = []
        for line in response.split('\n'):
            if line.strip().startswith(('1)', '2)', '3)', '4)')):
                steps.append(line.strip())
        return steps

    async def get_answer(self, query: str) -> Dict[str, Any]:
        # Retrieve relevant documents
        relevant_docs = await self.retriever.retrieve(query)
        
        # Combine context
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer using chain of thought
        response = await self.chain.arun({
            "question": query,
            "context": context
        })
        
        return {
            "answer": response,
            "sources": relevant_docs,
            "reasoning_chain": self._extract_reasoning(response)
        } 