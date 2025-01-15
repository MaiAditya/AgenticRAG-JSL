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
import fitz
import concurrent.futures

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
            # Open document using PyMuPDF
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            # Create semaphore for controlling concurrent processing
            semaphore = asyncio.Semaphore(self.parallel_processor.max_workers)
            
            async def process_page_with_semaphore(page_num):
                async with semaphore:
                    # Get the page
                    page = doc[page_num]
                    
                    # Create structure detection tool and analyze page
                    structure_tool = StructureDetectionTool()
                    # Since analyze_page is async, we need to await it
                    page_structure = await structure_tool.analyze_page(page)
                    
                    if "error" in page_structure:
                        logger.error(f"Error analyzing page {page_num}: {page_structure['error']}")
                        return {
                            'page_number': page_num,
                            'results': []
                        }
                    
                    page_results = []
                    tasks = []
                    
                    # Create extraction tasks for each component
                    for component in page_structure['components']:
                        if component['type'] == 'table':
                            tasks.append(self._extract_table(component, page_num))
                        elif component['type'] == 'text':
                            tasks.append(self._extract_text(component))
                        elif component['type'] == 'image':
                            tasks.append(self._extract_image(component))
                    
                    # Process all components concurrently
                    component_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Filter out errors and add successful results
                    for result in component_results:
                        if isinstance(result, dict) and not result.get('error'):
                            page_results.append(result)
                    
                    return {
                        'page_number': page_num,
                        'results': page_results
                    }
            
            # Create tasks for all pages
            tasks = [
                process_page_with_semaphore(page_num)
                for page_num in range(total_pages)
            ]
            
            # Process all pages concurrently
            all_results = await asyncio.gather(*tasks)
            
            # Combine results from all pages
            combined_results = []
            for page_result in all_results:
                combined_results.extend(page_result['results'])
            
            # Integrate knowledge after processing all pages
            if combined_results:
                try:
                    logger.info("Integrating extracted knowledge into vector store")
                    await self.knowledge_integrator.integrate_knowledge(combined_results)
                except Exception as e:
                    logger.error(f"Error during knowledge integration: {str(e)}")
            
            doc.close()
            
            return {
                'success': True,
                'processed_pages': total_pages,
                'results': combined_results
            }
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise ProcessingError(f"Failed to process document: {str(e)}")

    async def _extract_table(self, component, page_num):
        try:
            pix = component['image']
            extracted = await self.table_extractor.extract(pix)
            if extracted and 'table_data' in extracted and extracted['table_data']:
                return {
                    'type': 'table',
                    'page_number': page_num,
                    'extracted': extracted
                }
        except Exception as e:
            logger.error(f"Error extracting table: {str(e)}")
            return {'error': str(e)}

    async def _extract_text(self, component):
        try:
            extracted = await self.text_extractor.extract(component['content'])
            if extracted:
                return {
                    'type': 'text',
                    'content': component['content'],
                    'extracted': extracted
                }
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {'error': str(e)}

    async def _extract_image(self, component):
        try:
            extracted = await self.image_extractor.extract(component['image'])
            if not 'error' in extracted:
                return {
                    'type': 'image',
                    'location': component['location'],
                    'extracted': extracted
                }
        except Exception as e:
            logger.error(f"Error extracting image: {str(e)}")
            return {'error': str(e)}

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