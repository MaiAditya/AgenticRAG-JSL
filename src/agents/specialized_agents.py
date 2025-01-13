from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from typing import List, Dict, Any
from src.core.config import settings
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from datetime import datetime
from loguru import logger
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

class DocumentAnalysisAgent:
    """Agent responsible for analyzing document structure and coordinating extraction"""
    def __init__(self, tools):
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        system_message = SystemMessage(
            content="""You are an expert at analyzing documents and determining their structure.
            Identify key components like tables, images, and text sections. Plan the extraction strategy."""
        )
        
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=tools,
            system_message=system_message,
            input_variables=["input", "agent_scratchpad", "chat_history"]
        )
        
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

class ExtractionAgent:
    """Agent responsible for content extraction using various tools"""
    def __init__(self, extractors):
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.extractors = extractors
        
        system_message = SystemMessage(
            content="""You are an expert at extracting specific content from documents.
            Extract content based on the provided type and structure."""
        )
        
        tools = [self._create_tool(name, extractor) 
                for name, extractor in extractors.items()]
        
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=tools,
            system_message=system_message,
            input_variables=["input", "agent_scratchpad", "chat_history"]
        )
        
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
    
    def _create_tool(self, name: str, extractor: Any) -> BaseTool:
        tool_name = name  # Store name in closure
        
        class ExtractorTool(BaseTool):
            name = f"{tool_name}_extractor"
            description = f"Extract {tool_name} content from documents"
            extractor_instance = extractor
            
            async def _arun(self, content: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    logger.debug(f"Starting {tool_name} extraction")
                    
                    # Ensure content is properly formatted
                    if isinstance(content, dict):
                        extract_content = content.get('input', '')
                    else:
                        extract_content = str(content)
                    
                    # Ensure content is string
                    if not isinstance(extract_content, str):
                        extract_content = str(extract_content)
                    
                    result = await self.extractor_instance.extract(extract_content)
                    
                    # Ensure result is properly formatted
                    if not isinstance(result, dict):
                        result = {"content": str(result)}
                    
                    result.update({
                        "type": tool_name,
                        "success": True
                    })
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in {tool_name} extraction: {str(e)}")
                    return {
                        "type": tool_name,
                        "error": str(e),
                        "success": False
                    }
                    
            def _run(self, content: Dict[str, Any]) -> Dict[str, Any]:
                raise NotImplementedError("Use async version")
                
        return ExtractorTool()

class KnowledgeIntegrationAgent:
    """Agent responsible for integrating extracted information into knowledge base"""
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.vector_store = vector_store
        
        system_message = SystemMessage(
            content="""You are an expert at organizing and integrating information.
            Structure extracted content for optimal retrieval and understanding."""
        )
        
        self.tools = [self._create_integration_tools()]
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=system_message,
            input_variables=["input", "agent_scratchpad"],
            memory_prompts=[MessagesPlaceholder(variable_name="chat_history")]
        )
        
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )

    def _create_integration_tools(self):
        """Creates tools for knowledge base integration"""
        from langchain.tools import BaseTool
        
        class VectorStoreIntegrationTool(BaseTool):
            name = "vector_store_integration"
            description = "Integrate extracted content into the vector store"
            vector_store = self.vector_store
            
            async def _arun(self, content: Dict[str, Any]) -> bool:
                try:
                    # Extract metadata
                    metadata = {
                        "content_type": content.get("type", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "source": content.get("source", "unknown")
                    }
                    
                    # Process and store the content
                    text_content = str(content.get("content", ""))
                    
                    # Add to vector store
                    await self.vector_store.add_texts(
                        texts=[text_content],
                        metadatas=[metadata]
                    )
                    return True
                    
                except Exception as e:
                    logger.error(f"Error integrating content: {str(e)}")
                    return False
        
        return VectorStoreIntegrationTool() 