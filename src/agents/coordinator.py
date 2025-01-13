from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from .specialized_agents import DocumentAnalysisAgent, ExtractionAgent, KnowledgeIntegrationAgent
from loguru import logger
from src.core.config import settings
import asyncio
from datetime import datetime

class CoordinatorAgent:
    def __init__(self, tools: Dict[str, Any], vector_store):
        logger.info("Initializing CoordinatorAgent")
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.doc_analyzer = DocumentAnalysisAgent(tools["analysis"])
        self.extractor = ExtractionAgent(tools["extraction"])
        self.knowledge_integrator = KnowledgeIntegrationAgent(vector_store)
        logger.debug("CoordinatorAgent initialized successfully")

    def _create_extraction_tasks(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create extraction tasks based on document analysis"""
        tasks = []
        
        # Extract components from the analysis
        if 'components' in analysis:
            for component in analysis['components']:
                # Ensure content is properly converted to string
                content = component.get('content', '')
                if not isinstance(content, (str, list)):
                    content = str(content)
                    
                task = {
                    'content_type': component.get('type', 'text'),
                    'content': content,
                    'metadata': {
                        'type': component.get('type', 'text'),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                tasks.append(task)
        else:
            # Fallback to treating entire content as text
            content = analysis.get('content', '')
            if not isinstance(content, (str, list)):
                content = str(content)
                
            tasks.append({
                'content_type': 'text',
                'content': content,
                'metadata': {
                    'type': 'text',
                    'timestamp': datetime.now().isoformat()
                }
            })
        
        return tasks

    def combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple pages"""
        return {
            "pages": results,
            "total_pages": len(results),
            "timestamp": datetime.now().isoformat()
        }

    async def process_page(self, page) -> Dict[str, Any]:
        try:
            # Get page text content
            page_text = str(page.get_text())  # Ensure text is string
            
            # Analyze page structure
            analysis = await self.doc_analyzer.executor.ainvoke({
                "input": page_text,
                "chat_history": []
            })
            
            # Create extraction tasks
            tasks = self._create_extraction_tasks(analysis)
            
            # Process extractions
            extraction_results = []
            for task in tasks:
                try:
                    # Format the input properly for the extractor
                    formatted_input = {
                        "input": task['content'],
                        "type": task['content_type']
                    }
                    
                    result = await self.extractor.executor.ainvoke(formatted_input)
                    
                    # Add to vector store if extraction was successful
                    if result and not isinstance(result, dict):
                        result = {"content": str(result)}
                    
                    if result and "error" not in result:
                        try:
                            await self.knowledge_integrator.executor.ainvoke({
                                "content": str(result.get("content", "")),
                                "type": task['content_type'],
                                "source": f"page_{page.number}"
                            })
                        except Exception as e:
                            logger.error(f"Error integrating content: {str(e)}")
                    
                    extraction_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in extraction task for page {page.number}: {str(e)}")
                    extraction_results.append({
                        "error": str(e),
                        "success": False
                    })
            
            return {
                "page_num": page.number,
                "content": extraction_results,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "num_extractions": len(extraction_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page.number}: {str(e)}")
            raise

    async def _execute_parallel_extraction(self, tasks: List[Dict]) -> List[Dict]:
        # Execute extraction tasks in parallel
        async with asyncio.TaskGroup() as group:
            extraction_tasks = [
                group.create_task(self.extractor.executor.arun(task))
                for task in tasks
            ]
        return [task.result() for task in extraction_tasks] 