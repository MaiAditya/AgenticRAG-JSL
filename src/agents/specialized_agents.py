from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from loguru import logger
from src.vectorstore.chroma_store import ChromaStore
from langchain_openai import ChatOpenAI
from src.core.config import settings
from enum import Enum

class SupportedFileTypes(Enum):
    PDF = "pdf"
    TXT = "txt"
    DOC = "doc"
    DOCX = "docx"
    TEXT = "text"  # Add support for plain text content
    UNKNOWN = "unknown"  # Default type for text content

    @classmethod
    def get_type(cls, content_or_path: str) -> str:
        """Get file type from content or path"""
        if not content_or_path:
            return cls.UNKNOWN.value
            
        if isinstance(content_or_path, str) and '.' in content_or_path:
            ext = content_or_path.split('.')[-1].lower()
            try:
                return cls(ext).value
            except ValueError:
                return cls.TEXT.value
        return cls.TEXT.value

class DocumentAnalysisAgent:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-1106-preview",
            api_key=settings.OPENAI_API_KEY
        )
        logger.info("Initializing DocumentAnalysisAgent")

    async def analyze_page(self, page_text: str) -> Dict[str, Any]:
        """Analyze the structure of a page and identify components"""
        try:
            # Use direct LLM call instead of agent for simpler analysis
            prompt = f"""Analyze the following document page content and identify its components.
            Identify sections like:
            - Regular text paragraphs
            - Tables (if any tabular data is detected)
            - Lists or bullet points
            - Headers or titles
            
            Return the components in a structured format.
            
            Page content:
            {page_text}
            """
            
            response = await self.llm.ainvoke(prompt)
            
            # Process the response into components
            components = [
                {"type": "text", "content": page_text}  # Default fallback
            ]
            
            if response.content:
                # Add any identified components from the analysis
                components = self._parse_analysis(response.content)
            
            return {
                "components": components
            }
        except Exception as e:
            logger.error(f"Error analyzing page: {str(e)}")
            return {"error": str(e)}
            
    def _parse_analysis(self, analysis: str) -> List[Dict[str, Any]]:
        """Parse the LLM analysis into structured components"""
        components = []
        try:
            # Simple parsing - treat each line as a potential component
            current_component = {"type": "text", "content": ""}
            
            for line in analysis.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Map all components to text type for now
                # Headers
                if line.isupper() or line.endswith(':'):
                    if current_component["content"]:
                        components.append(current_component)
                    current_component = {"type": "text", "content": line}
                # Lists
                elif line.startswith(('•', '-', '*')) or line[0].isdigit():
                    if current_component["content"]:
                        components.append(current_component)
                    current_component = {"type": "text", "content": line}
                # Tables (keep as is)
                elif line.startswith(('Table:', 'TABLE')):
                    if current_component["content"]:
                        components.append(current_component)
                    current_component = {"type": "table", "content": line}
                # Regular text
                else:
                    if current_component["type"] != "text":
                        components.append(current_component)
                        current_component = {"type": "text", "content": line}
                    else:
                        current_component["content"] += "\n" + line
            
            if current_component["content"]:
                components.append(current_component)
                
        except Exception as e:
            logger.error(f"Error parsing analysis: {str(e)}")
            components = [{"type": "text", "content": analysis}]
            
        return components if components else [{"type": "text", "content": analysis}]

    def validate_file_type(self, file_path: str) -> Optional[str]:
        file_extension = file_path.split('.')[-1].lower()
        try:
            return SupportedFileTypes(file_extension).value
        except ValueError:
            logger.error(f"Unsupported file type: {file_extension}")
            return None
            
    async def extract_component(self, file_path: str, content: Any) -> Dict[str, Any]:
        file_type = self.validate_file_type(file_path)
        if not file_type:
            raise ValueError(f"Unsupported file type for: {file_path}")
            
        extractor = self.extractors.get(file_type)
        if not extractor:
            raise ValueError(f"No extractor found for file type: {file_type}")
            
        return await extractor.extract(content)

class ExtractionAgent:
    def __init__(self, extractors: Dict[str, Any]):
        self.extractors = extractors
        logger.info("Initializing ExtractionAgent")

    async def extract_component(self, component: Dict[str, Any], page_text: str = None) -> Dict[str, Any]:
        """Extract content from a specific component"""
        try:
            component_type = component.get("type", "text")
            content = component.get("content", "")
            
            # Always use text extractor for component extraction
            extractor = self.extractors.get("text")
            if not extractor:
                raise ValueError("Text extractor not found")
            
            result = await extractor.extract(content)
            return {
                "type": component_type,
                "content": content,
                "extracted": result
            }
            
        except Exception as e:
            logger.error(f"Error extracting component: {str(e)}")
            return {
                "type": "text",
                "content": component.get("content", ""),
                "error": str(e)
            }

class KnowledgeIntegrationAgent:
    def __init__(self, vector_store: ChromaStore):
        self.vector_store = vector_store
        logger.info("Initializing KnowledgeIntegrationAgent")

    async def integrate_knowledge(self, results: List[Dict[str, Any]]) -> None:
        """Integrate extracted knowledge into the vector store"""
        try:
            # Process and store the extracted information
            documents = []
            metadatas = []
            
            for result in results:
                if result.get('type') == 'text' and result.get('content'):
                    documents.append(result['content'])
                    metadatas.append({
                        'type': 'text',
                        'source': 'document'
                    })
                elif result.get('type') == 'table' and result.get('extracted'):
                    table_text = str(result['extracted'].get('table_data', ''))
                    if table_text:
                        documents.append(table_text)
                        metadatas.append({
                            'type': 'table',
                            'source': 'document'
                        })
            
            if documents:
                logger.info(f"Adding {len(documents)} documents to vector store")
                await self.vector_store.add_texts(documents, metadatas)
                logger.info(f"Successfully integrated {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error integrating knowledge: {str(e)}")
            raise