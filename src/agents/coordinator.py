from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
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
            
            # Create a thread pool for CPU-bound operations
            thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.parallel_processor.max_workers
            )
            
            async def process_page_with_semaphore(page_num):
                async with semaphore:
                    page = doc[page_num]
                    
                    # Run CPU-bound operations in thread pool
                    loop = asyncio.get_event_loop()
                    
                    # Process text extraction in thread pool
                    text_content = await loop.run_in_executor(
                        thread_pool, 
                        page.get_text
                    )
                    
                    # Create structure detection tool
                    structure_tool = StructureDetectionTool()
                    
                    # Process components in parallel
                    components = []
                    if text_content.strip():
                        components.append({
                            "type": "text",
                            "content": text_content
                        })
                    
                    # Get pixmap for table detection
                    pix = await loop.run_in_executor(
                        thread_pool,
                        page.get_pixmap
                    )
                    
                    if pix:
                        # Process tables with descriptions
                        if self.table_extractor:
                            table_result = await self.table_extractor.extract(pix)
                            if table_result and "table_data" in table_result:
                                for table in table_result["table_data"]:
                                    components.append({
                                        "type": "table",
                                        "content": table["structure"],
                                        "description": table.get("description", ""),
                                        "metadata": {
                                            "confidence": table["confidence"],
                                            "num_rows": table["structure"]["num_rows"],
                                            "num_cols": table["structure"]["num_cols"]
                                        }
                                    })
                    
                    # Get images in parallel
                    image_list = await loop.run_in_executor(
                        thread_pool,
                        page.get_images
                    )
                    
                    for img in image_list:
                        components.append({
                            "type": "image",
                            "location": img,
                            "image": await loop.run_in_executor(thread_pool, page.get_pixmap)
                        })
                    
                    # Process all components concurrently
                    extraction_tasks = []
                    for component in components:
                        if component['type'] == 'table':
                            extraction_tasks.append(self._extract_table(component, page_num))
                        elif component['type'] == 'text':
                            extraction_tasks.append(self._extract_text(component))
                        elif component['type'] == 'image':
                            extraction_tasks.append(self._extract_image(component))
                    
                    # Execute all extraction tasks concurrently
                    component_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                    
                    # Filter successful results
                    page_results = [
                        result for result in component_results 
                        if isinstance(result, dict) and not result.get('error')
                    ]
                    
                    return {
                        'page_number': page_num,
                        'results': page_results
                    }
            
            # Process all pages concurrently
            tasks = [
                asyncio.create_task(process_page_with_semaphore(page_num))
                for page_num in range(total_pages)
            ]
            
            # Wait for all pages to complete
            all_results = await asyncio.gather(*tasks)
            
            # Combine results
            combined_results = []
            for page_result in all_results:
                combined_results.extend(page_result['results'])
            
            # Integrate knowledge in parallel if results exist
            if combined_results:
                try:
                    logger.info("Integrating extracted knowledge into vector store")
                    await self.knowledge_integrator.integrate_knowledge(combined_results)
                except Exception as e:
                    logger.error(f"Error during knowledge integration: {str(e)}")
            
            # Cleanup
            thread_pool.shutdown(wait=False)
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
            if 'content' not in component:
                logger.error("Table component missing content")
                return {'error': 'Missing table content'}
            
            table_data = component['content']
            description = component.get('description', '')
            metadata = component.get('metadata', {})
            
            # Prepare text representation for vector storage
            text_representation = f"""
            Table Description: {description}
            Number of Rows: {metadata.get('num_rows', 'Unknown')}
            Number of Columns: {metadata.get('num_cols', 'Unknown')}
            Confidence: {metadata.get('confidence', 'Unknown')}
            Table Content: {json.dumps(table_data, indent=2)}
            """
            
            return {
                'type': 'table',
                'page_number': page_num,
                'text_content': text_representation,
                'structured_data': table_data,
                'description': description,
                'metadata': metadata
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
                # Prepare text representation for vector storage
                text_representation = f"""
                Visual Element Type: {extracted['visual_type']}
                Description: {extracted['description']}
                Extracted Text: {', '.join(extracted['extracted_text'])}
                """
                
                # Store both text and structured data
                return {
                    'type': 'visual_element',
                    'text_content': text_representation,
                    'structured_data': extracted,
                    'metadata': {
                        'visual_type': extracted['visual_type'],
                        'element_count': len(extracted.get('elements', [])),
                        'has_text': bool(extracted.get('extracted_text'))
                    }
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