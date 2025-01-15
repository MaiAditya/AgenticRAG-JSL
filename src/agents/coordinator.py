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
            logger.info(f"Starting document processing for: {file_path}")
            results = []
            
            # Detect content type and structure
            structure_tool = StructureDetectionTool()
            document_structure = await structure_tool._run(file_path)
            
            logger.info(f"Found {len(document_structure['components'])} components to process")
            
            for component in document_structure['components']:
                try:
                    if component['type'] == 'table':
                        logger.info(f"Processing table component from page {component.get('page_number')}")
                        if self.table_extractor and component.get('raw_image'):
                            # Convert raw image bytes to format expected by table extractor
                            table_image = Image.open(io.BytesIO(component['raw_image']))
                            extracted = await self.table_extractor.extract(table_image)
                            
                            if extracted and 'table_data' in extracted:
                                results.append({
                                    'type': 'table',
                                    'page_number': component.get('page_number'),
                                    'extracted': extracted,
                                    'metadata': extracted.get('metadata', {})
                                })
                                logger.info(f"Successfully extracted table from page {component.get('page_number')}")
                    
                    elif component['type'] == 'text':
                        logger.info(f"Processing text component from page {component.get('page_number')}")
                        if self.text_extractor:
                            extracted = await self.text_extractor.extract(component['content'])
                            if extracted and 'content' in extracted:
                                results.append({
                                    'type': 'text',
                                    'content': extracted['content'],
                                    'elements': extracted.get('elements', []),
                                    'page_number': component.get('page_number')
                                })
                                logger.info("Successfully extracted text content")
                    
                    elif component['type'] == 'image':
                        logger.info(f"Processing image component from page {component.get('metadata', {}).get('page_number')}")
                        if self.image_extractor and component.get('image'):
                            image_data = component['image']
                            extracted = await self.image_extractor.extract({
                                'image': Image.open(io.BytesIO(image_data)),
                                'metadata': component.get('metadata', {})
                            })
                            
                            if extracted and 'error' not in extracted:
                                results.append({
                                    'type': 'image',
                                    'page_number': component.get('metadata', {}).get('page_number'),
                                    'extracted': extracted
                                })
                                logger.info(f"Successfully extracted image content")
                    
                except Exception as component_error:
                    logger.error(f"Error processing component: {str(component_error)}")
                    continue

            # Integrate knowledge after processing all components
            if results:
                try:
                    logger.info("Integrating extracted knowledge into vector store")
                    await self.knowledge_integrator.integrate_knowledge(results)
                except Exception as e:
                    logger.error(f"Error during knowledge integration: {str(e)}")
            
            final_result = {
                'success': True,
                'processed_pages': document_structure.get('processed_pages', 0),
                'results': results,
                'statistics': {
                    'total_components': len(document_structure['components']),
                    'processed_components': len(results),
                    'component_types': {
                        'text': len([r for r in results if r['type'] == 'text']),
                        'table': len([r for r in results if r['type'] == 'table']),
                        'image': len([r for r in results if r['type'] == 'image'])
                    }
                }
            }
            
            logger.info(f"Document processing completed with statistics: {final_result['statistics']}")
            return final_result
            
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

    async def combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                await self.knowledge_integrator.integrate_knowledge(combined["results"])
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