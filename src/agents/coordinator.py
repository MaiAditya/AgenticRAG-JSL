from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from .specialized_agents import DocumentAnalysisAgent, ExtractionAgent, KnowledgeIntegrationAgent
from loguru import logger
from src.core.config import settings
import asyncio
from datetime import datetime
import json
from langchain.tools import BaseTool
from src.vectorstore.chroma_store import ChromaStore
from src.core.cache import DocumentCache
from src.core.types import DocumentProcessor
from src.processors.parallel_processor import ParallelDocumentProcessor
from src.tools.document_tools import StructureDetectionTool, ContentExtractionTool
from src.extractors.text_extractor import TextExtractor
from src.extractors.table_extractor import TableExtractor
from src.extractors.image_extractor import ImageExtractor
from src.core.error_handling import ProcessingError
import io
from PIL import Image

class CoordinatorAgent(DocumentProcessor):
    def __init__(self, extractors: Dict[str, Any], tools: List[BaseTool], 
                 vector_store: ChromaStore, document_cache: DocumentCache):
        self.extractors = extractors
        self.doc_analyzer = DocumentAnalysisAgent(tools=tools)
        self.extractor = ExtractionAgent(extractors=extractors)
        self.knowledge_integrator = KnowledgeIntegrationAgent(vector_store=vector_store)
        self.document_cache = document_cache
        self.parallel_processor = ParallelDocumentProcessor(self)
        # Use extractors passed in instead of creating new instances
        self.text_extractor = extractors.get('text')
        self.table_extractor = extractors.get('table')
        self.image_extractor = extractors.get('image')
        logger.info("Initializing CoordinatorAgent")

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            results = []
            
            # Detect content type and structure
            structure_tool = StructureDetectionTool()
            document_structure = await structure_tool.analyze(file_path)
            
            for component in document_structure['components']:
                try:
                    if component['type'] == 'table':
                        logger.info("Processing table component")
                        # Convert pixmap to PIL Image
                        pix = component['image']
                        try:
                            # Send the pixmap directly to the table extractor
                            extracted = await self.table_extractor.extract(pix)
                            if extracted and 'table_data' in extracted and extracted['table_data']:
                                results.append({
                                    'type': 'table',
                                    'page_number': component.get('page_number'),
                                    'extracted': extracted
                                })
                                logger.info(f"Successfully extracted table from page {component.get('page_number')}")
                        except Exception as img_error:
                            logger.error(f"Error converting image for table extraction: {str(img_error)}")
                            continue
                    elif component['type'] == 'text':
                        extracted = await self.text_extractor.extract(component['content'])
                        results.append({
                            'type': 'text',
                            'content': component['content'],
                            'extracted': extracted
                        })
                    elif component['type'] == 'image':
                        extracted = await self.image_extractor.extract(component['image'])
                        if 'error' in extracted:
                            logger.warning(f"Image extraction warning: {extracted['error']}")
                            continue
                        results.append({
                            'type': 'image',
                            'location': component['location'],
                            'extracted': extracted
                        })
                except Exception as component_error:
                    logger.error(f"Error processing component: {str(component_error)}")
                    continue
            
            return {
                'success': True,
                'processed_pages': document_structure['processed_pages'],
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise ProcessingError(f"Failed to process document: {str(e)}")

    async def process_page(self, page) -> Dict[str, Any]:
        """Process a single page of the document"""
        try:
            # Extract page content
            page_text = page.get_text()
            
            # Analyze page structure using DocumentAnalysisTool
            analysis = await self.doc_analyzer.analyze_page(page_text)
            
            if "error" in analysis:
                logger.error(f"Error in document analysis: {analysis['error']}")
                return {
                    "success": False,
                    "error": analysis['error'],
                    "page_number": page.number
                }
            
            extraction_results = []
            
            # Process each component using appropriate extractor
            for component in analysis.get("components", []):
                try:
                    # Pass only component and page_text
                    result = await self.extractor.extract_component(component, page_text)
                    if result:
                        extraction_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing component: {str(e)}")
            
            return {
                "success": True,
                "page_number": page.number,
                "results": extraction_results
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page.number}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "page_number": page.number
            }

    def combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple pages"""
        combined = {
            "success": all(r.get("success", False) for r in results),
            "total_pages": len(results),
            "processed_pages": len([r for r in results if r.get("success", False)]),
            "results": []
        }
        
        for result in results:
            if result.get("success"):
                combined["results"].extend(result.get("results", []))
        
        # Integrate knowledge if results are successful
        if combined["success"] and combined["results"]:
            try:
                self.knowledge_integrator.integrate_knowledge(combined["results"])
            except Exception as e:
                logger.error(f"Error integrating knowledge: {str(e)}")
        
        return combined

    async def _execute_parallel_extraction(self, tasks: List[Dict]) -> List[Dict]:
        # Execute extraction tasks in parallel
        async with asyncio.TaskGroup() as group:
            extraction_tasks = [
                group.create_task(self.extractor.executor.arun(task))
                for task in tasks
            ]
        return [task.result() for task in extraction_tasks] 