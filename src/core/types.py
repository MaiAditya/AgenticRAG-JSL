from typing import Protocol, Dict, Any, List, TypedDict, Union

class PageResult(TypedDict):
    success: bool
    page_number: int
    results: List[Dict[str, Any]]
    error: Union[str, None]

class CombinedResult(TypedDict):
    success: bool
    total_pages: int
    processed_pages: int
    results: List[Dict[str, Any]]

class DocumentProcessor(Protocol):
    """Protocol defining the interface for document processing"""
    
    async def process_page(self, page) -> PageResult:
        """Process a single page of the document
        
        Args:
            page: The page object from PyMuPDF (fitz)
            
        Returns:
            PageResult containing the extraction results or error information
        """
        ...
    
    def combine_results(self, results: List[PageResult]) -> CombinedResult:
        """Combine results from multiple pages
        
        Args:
            results: List of PageResults from individual pages
            
        Returns:
            CombinedResult containing all processed information
        """
        ... 