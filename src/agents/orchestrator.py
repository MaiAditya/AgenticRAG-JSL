from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool
from typing import List

class OrchestratorAgent:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    async def process_query(self, query: str) -> str:
        response = await self.agent_executor.arun(query)
        return response 