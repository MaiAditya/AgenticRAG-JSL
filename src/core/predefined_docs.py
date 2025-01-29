from pathlib import Path
import json
import os
from loguru import logger
from typing import Dict, Any, List
import aiofiles
import asyncio
from src.vectorstore.chroma_store import ChromaStore
from src.core.cache import DocumentCache
from datetime import datetime

class PredefinedDocsManager:
    def __init__(self, vector_store: ChromaStore, document_cache: DocumentCache):
        self.vector_store = vector_store
        self.document_cache = document_cache
        self.predefined_dir = Path("predefined-pdfs")
        self.cache_dir = Path("predefined-cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    async def initialize_predefined_docs(self):
        """Initialize all predefined documents in the vector store"""
        try:
            logger.info("Initializing predefined documents...")
            
            # Get all predefined PDFs
            all_pdfs = list(self.predefined_dir.glob("*.pdf"))
            processed_docs = []
            
            # Process each PDF
            for pdf_path in all_pdfs:
                logger.info(f"Loading document: {pdf_path}")
                cache_path = self.cache_dir / f"{pdf_path.stem}_results.json"
                
                if cache_path.exists():
                    # Load from cache and add to vector store
                    async with aiofiles.open(cache_path, 'r') as f:
                        content = await f.read()
                        cached_results = json.loads(content)
                        
                        if 'texts' in cached_results:
                            # Ensure valid metadata
                            metadatas = cached_results.get('metadatas', [])
                            if not metadatas or not all(metadatas):
                                metadatas = [{
                                    'source': str(pdf_path),
                                    'type': 'predefined_document',
                                    'timestamp': datetime.now().isoformat(),
                                    'filename': pdf_path.name
                                } for _ in cached_results['texts']]
                            
                            # Always add to vector store on startup
                            await self.vector_store.add_texts(
                                texts=cached_results['texts'],
                                metadatas=metadatas
                            )
                            processed_docs.append(str(pdf_path))
                else:
                    # Only process if not cached
                    logger.info(f"Processing new document: {pdf_path}")
                    with pdf_path.open('rb') as file:
                        result = await self.process_document(file)
                        if result and 'results' in result:
                            texts = []
                            metadatas = []
                            
                            for item in result['results']:
                                if 'content' in item:
                                    texts.append(item['content'])
                                    metadatas.append({
                                        'source': str(pdf_path),
                                        'type': 'predefined_document',
                                        'content_type': item.get('type', 'text'),
                                        'timestamp': datetime.now().isoformat(),
                                        'filename': pdf_path.name
                                    })
                            
                            if texts:
                                await self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                                
                                # Cache the results
                                cache_data = {
                                    'texts': texts,
                                    'metadatas': metadatas
                                }
                                async with aiofiles.open(cache_path, 'w') as f:
                                    await f.write(json.dumps(cache_data))
                                
                                processed_docs.append(str(pdf_path))
            
            # Update processed docs cache
            cache_file = self.cache_dir / "processed_docs.json"
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(processed_docs))
                
            logger.info(f"Predefined documents initialization complete. Processed {len(processed_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing predefined documents: {str(e)}")
            raise

    async def process_document(self, file) -> Dict[str, Any]:
        """Process a single document and return its results"""
        try:
            # Check if file is a path or file object
            file_path = file if isinstance(file, str) else file.name
            cache_path = self.cache_dir / f"{Path(file_path).stem}_results.json"
            
            # If cached, return cached results
            if cache_path.exists():
                logger.info(f"Loading cached results for {file_path}")
                async with aiofiles.open(cache_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            
            # If not cached, process normally
            from src.agents.coordinator import CoordinatorAgent
            from src.core.initialization import get_extractors, get_tools
            
            extractors = get_extractors()
            tools = get_tools()
            
            coordinator = CoordinatorAgent(
                extractors=extractors,
                tools=tools,
                vector_store=self.vector_store,
                document_cache=self.document_cache
            )
            
            result = await coordinator.process_document(file)
            
            if result:
                # Cache the results
                async with aiofiles.open(cache_path, 'w') as f:
                    await f.write(json.dumps(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None 