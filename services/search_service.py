"""
Academic Paper Search Service

This module provides search capabilities for academic papers using the arXiv API.
It handles query processing, result filtering, metadata extraction, and paper
object formatting for integration with the research system.

Classes:
    SearchService: Main service class for academic paper search operations

Features:
    - arXiv API integration with search capabilities
    - Date range filtering and relevance sorting
    - Paper metadata extraction
    - Error handling and graceful degradation
    - Configurable result limits and search parameters

Dependencies:
    - arxiv: Python library for arXiv API access
    - system.config: Configuration management
    - datetime: Date handling utilities
"""

import arxiv
from typing import List, Dict, Any, Optional
from datetime import datetime
from system.config import Config


class SearchService:
    """
    Academic paper search service using arXiv API.
    
    Provides search capabilities for academic papers with filtering, metadata 
    extraction, and result formatting. Handles query processing, applies 
    user-defined filters, and converts arXiv results into paper objects.
    
    Attributes:
        config: arXiv-specific configuration settings including limits and preferences
    
    Methods:
        search_arxiv: Main search method with filtering and result processing
        
    Private Methods:
        _format_arxiv_result: Convert arXiv results to paper objects
        _apply_date_filters: Apply publication date range filtering
        _validate_search_parameters: Validate and sanitize search inputs
        _generate_paper_citation: Create citation strings
        _extract_paper_metadata: Extract metadata from results
    """
    
    def __init__(self):
        """
        Initialize the search service with arXiv API configuration.
        
        Sets up the service with appropriate limits, sorting criteria,
        and other search parameters from the system configuration.
        """
        self.config = Config.get_arxiv_config()
    
    def search_arxiv(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search arXiv for academic papers matching the specified query and filters.
        
        Performs a comprehensive search of the arXiv database with support for
        advanced filtering, date ranges, and result limits. Handles errors gracefully
        and returns standardized paper objects for system integration.
        
        Args:
            query: Search query string using arXiv search syntax
            filters: Optional dictionary containing:
                - date_range: Tuple of (start_year, end_year) for publication filtering
                - max_results: Maximum number of results to return
                - sort_criteria: Sorting preference for results
                
        Returns:
            List of standardized paper dictionaries containing:
                - id: Unique paper identifier
                - title: Paper title
                - authors: List of author names
                - abstract: Paper abstract/summary
                - url: Link to PDF or paper page
                - published_date: Publication date
                - source: Source identifier ("arXiv")
                - citation: Formatted citation string
                - content: Full text and structured sections
                - metadata: Additional arXiv-specific information
                
        Raises:
            No exceptions - all errors are handled gracefully with empty result list
        """
        try:
            # Validate and prepare search parameters
            search_params = self._validate_search_parameters(query, filters)
            
            # Execute arXiv search
            papers = self._execute_arxiv_search(search_params)
            
            return papers
            
        except Exception as e:
            return []
    
    def _format_arxiv_result(self, result) -> Dict[str, Any]:
        """
        Convert arXiv search result to standardized paper object format.
        
        Extracts and formats all relevant metadata from an arXiv result object
        into the system's standardized paper format. Includes comprehensive
        metadata extraction, citation generation, and content structuring.
        
        Args:
            result: arXiv search result object from the arxiv library
            
        Returns:
            Standardized paper dictionary with complete metadata and content structure
        """
        # Extract basic identifiers and metadata
        arxiv_id = result.entry_id.split('/')[-1]
        paper_id = f"arxiv_{arxiv_id}"
        
        # Process author information
        authors = [author.name for author in result.authors]
        
        # Format publication information
        published_date = result.published.strftime("%Y-%m-%d")
        publication_year = result.published.year
        
        # Generate citation and content structure
        citation = self._generate_paper_citation(result, authors, publication_year, arxiv_id)
        metadata = self._extract_paper_metadata(result, arxiv_id)
        
        return {
            "id": paper_id,
            "title": result.title,
            "authors": authors,
            "abstract": result.summary,
            "url": result.pdf_url,
            "published_date": published_date,
            "source": "arXiv",
            "citation": citation,
            "content": {
                "full_text": result.summary,  # Use abstract as initial content
                "sections": {
                    "abstract": result.summary
                }
            },
            "metadata": metadata
        }
    
    def _validate_search_parameters(self, query: str, filters: Optional[Dict]) -> Dict[str, Any]:
        """
        Validate and prepare search parameters for arXiv query.
        
        Args:
            query: Raw search query
            filters: Optional filter dictionary
            
        Returns:
            Validated and prepared search parameters
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        # Determine result limit
        max_results = self.config["max_results"]
        if filters and filters.get("max_results"):
            max_results = min(filters["max_results"], max_results)
        
        return {
            "query": query.strip(),
            "max_results": max_results,
            "filters": filters or {}
        }
    
    def _execute_arxiv_search(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute the actual arXiv search with processing and filtering.
        
        Args:
            search_params: Validated search parameters
            
        Returns:
            List of processed paper objects
        """
        # Create arXiv search object
        search = arxiv.Search(
            query=search_params["query"],
            max_results=search_params["max_results"],
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            try:
                # Apply date filtering if specified
                if not self._apply_date_filters(result, search_params["filters"]):
                    continue
                
                # Format and add paper
                paper = self._format_arxiv_result(result)
                papers.append(paper)
                
            except Exception as e:
                continue
        
        return papers
    
    def _apply_date_filters(self, result, filters: Dict) -> bool:
        """
        Apply publication date filtering to search results.
        
        Args:
            result: arXiv result object
            filters: Filter dictionary containing date_range
            
        Returns:
            True if paper passes date filter, False otherwise
        """
        if not filters or "date_range" not in filters:
            return True
        
        date_range = filters["date_range"]
        paper_year = result.published.year
        
        return date_range[0] <= paper_year <= date_range[1]
    
    def _generate_paper_citation(self, result, authors: List[str], year: int, arxiv_id: str) -> str:
        """
        Generate a standardized citation string for the paper.
        
        Args:
            result: arXiv result object
            authors: List of author names
            year: Publication year
            arxiv_id: arXiv identifier
            
        Returns:
            Formatted citation string
        """
        authors_str = ", ".join(authors)
        return f"{authors_str} ({year}). {result.title}. arXiv:{arxiv_id}"
    
    def _extract_paper_metadata(self, result, arxiv_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from arXiv result.
        
        Args:
            result: arXiv result object
            arxiv_id: arXiv identifier
            
        Returns:
            Dictionary containing arXiv-specific metadata
        """
        return {
            "arxiv_id": arxiv_id,
            "categories": [cat for cat in result.categories],
            "primary_category": result.primary_category,
            "entry_id": result.entry_id,
            "updated": result.updated.isoformat() if result.updated else None
        }
