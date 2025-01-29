from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.tools import BaseTool
from typing import List
from langchain_openai import ChatOpenAI
from src.core.config import settings
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder

class OrchestratorAgent:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        system_message = SystemMessage(
            content="You are an expert orchestrator that coordinates complex document processing tasks."
        )
        
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=system_message,
            input_variables=["input", "agent_scratchpad"],
            memory_prompts=[MessagesPlaceholder(variable_name="chat_history")]
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    async def process_query(self, query: str) -> str:
        response = await self.agent_executor.arun(query)
        return response 