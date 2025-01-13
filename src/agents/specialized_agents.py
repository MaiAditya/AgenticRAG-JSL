from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_structured_chat_agent, AgentExecutor
from typing import List, Dict, Any

class DocumentAnalysisAgent:
    """Agent responsible for analyzing document structure and coordinating extraction"""
    def __init__(self, tools):
        self.llm = ChatOpenAI(temperature=0)
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=tools,
            system_message="""You are an expert at analyzing documents and determining their structure.
            Identify key components like tables, images, and text sections. Plan the extraction strategy."""
        )
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=True
        )

class ExtractionAgent:
    """Agent responsible for content extraction using various tools"""
    def __init__(self, extractors):
        self.llm = ChatOpenAI(temperature=0)
        self.extractors = extractors
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=[self._create_tool(name, extractor) for name, extractor in extractors.items()],
            system_message="""You are an expert at extracting content from documents.
            Use the appropriate tools to extract text, tables, and images accurately."""
        )
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

class KnowledgeIntegrationAgent:
    """Agent responsible for integrating extracted information into knowledge base"""
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(temperature=0)
        self.vector_store = vector_store
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=[self._create_integration_tools()],
            system_message="""You are an expert at organizing and integrating information.
            Structure extracted content for optimal retrieval and understanding."""
        ) 