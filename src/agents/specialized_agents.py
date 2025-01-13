from typing import Dict, Any, List
from langchain.tools import BaseTool
from loguru import logger
from src.vectorstore.chroma_store import ChromaStore
from langchain_openai import ChatOpenAI
from src.core.config import settings

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

class ExtractionAgent:
    def __init__(self, extractors: Dict[str, Any]):
        self.extractors = extractors
        logger.info("Initializing ExtractionAgent")

    async def extract_component(self, component: Dict[str, Any], page_text: str = None) -> Dict[str, Any]:
        """Extract content from a specific component"""
        try:
            component_type = component.get("type", "text")  # Default to text type
            content = component.get("content", "")
            
            # Map unknown types to text
            if component_type not in self.extractors:
                logger.warning(f"No extractor found for component type: {component_type}, defaulting to text")
                component_type = "text"
                
            extractor = self.extractors.get(component_type)
            if not extractor:
                raise ValueError(f"No extractor found for component type: {component_type}")
            
            # Pass both content and page_text to extractor if needed
            result = await extractor.extract(content, page_text) if page_text else await extractor.extract(content)
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

    def integrate_knowledge(self, results: List[Dict[str, Any]]) -> None:
        """Integrate extracted knowledge into the vector store"""
        try:
            # Process and store the extracted information
            documents = []
            for result in results:
                if isinstance(result.get("extracted_data"), str):
                    documents.append(result["extracted_data"])
            
            if documents:
                self.vector_store.add_texts(documents)
                logger.info(f"Successfully integrated {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error integrating knowledge: {str(e)}")
            raise