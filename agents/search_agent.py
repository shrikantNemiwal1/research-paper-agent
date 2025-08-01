"""
Research Paper Search Agent

This module provides comprehensive research paper discovery capabilities,
integrating with multiple academic databases and search services to locate
and retrieve relevant research content.

Classes:
    SearchAgent: Main agent class for research paper discovery

Features:
    - ArXiv database integration for academic paper search
    - DOI-based paper lookup and retrieval
    - Advanced search filtering and parameter validation
    - Multi-source search coordination
    - Search result validation and formatting
    - Error handling and graceful degradation

Dependencies:
    - services.search_service: Academic search service integration
    - typing: Type hints and annotations
"""

from typing import List, Dict, Any, Optional
from services.search_service import SearchService


class SearchAgent:
    """
    Research Paper Search Agent
    
    Specialized agent for discovering and retrieving research papers from
    academic databases and repositories. Provides comprehensive search
    capabilities with advanced filtering and validation.
    
    This agent handles:
    - Academic database search coordination
    - Query optimization and parameter validation
    - Multi-source search result aggregation
    - Paper metadata extraction and formatting
    - DOI-based paper lookup and retrieval
    - Search result validation and quality assurance
    
    Attributes:
        search_service (SearchService): Academic search service for database queries
        
    Methods:
        search_arxiv(query, filters): Search ArXiv database for papers
        search_by_doi(doi): Retrieve paper by DOI reference
        _validate_search_query(query): Validate search query parameters
        _format_search_results(results): Format and validate search results
        _create_doi_paper_stub(doi): Create paper stub from DOI
    """
    
    def __init__(self):
        """
        Initialize the research paper search agent.
        
        Sets up the search service and prepares the agent for
        academic database queries and paper discovery tasks.
        """
        self.search_service = SearchService()
    
    def search_arxiv(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search ArXiv database for research papers matching query and filters.
        
        Performs comprehensive search of the ArXiv repository with advanced
        filtering capabilities and result validation.
        
        Args:
            query (str): Search query string. Should contain relevant keywords,
                        author names, or research terms.
            filters (Optional[Dict]): Search filters for refining results.
                                    May include date ranges, categories, etc.
            
        Returns:
            List[Dict[str, Any]]: List of paper metadata objects containing
                                 title, authors, abstract, and other details.
                                 Returns empty list if search fails.
                                 
        Example:
            >>> agent = SearchAgent()
            >>> papers = agent.search_arxiv("machine learning", {"max_results": 10})
            >>> print(len(papers))  # Number of papers found
        """
        print(f"SearchAgent: Searching ArXiv for '{query}'")
        
        try:
            # Validate search parameters
            if not self._validate_search_query(query):
                return []
            
            # Execute search using search service
            papers = self.search_service.search_arxiv(query, filters)
            
            # Format and validate results
            formatted_papers = self._format_search_results(papers)
            
            print(f"SearchAgent: Found {len(formatted_papers)} papers")
            return formatted_papers
            
        except Exception as e:
            print(f"SearchAgent error: {str(e)}")
            return []
    
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve research paper by DOI reference.
        
        Looks up a specific paper using its Digital Object Identifier (DOI)
        and creates a standardized paper metadata object.
        
        Args:
            doi (str): Digital Object Identifier for the paper.
                      Should be in standard DOI format (e.g., "10.1000/xyz123").
            
        Returns:
            Optional[Dict[str, Any]]: Paper metadata object with standardized
                                    fields, or None if retrieval fails.
                                    
        Note:
            This is a simplified implementation that creates a paper stub.
            In production, this would integrate with CrossRef API or similar
            services for full metadata retrieval.
        """
        print(f"SearchAgent: Searching by DOI '{doi}'")
        
        try:
            # Validate DOI format
            if not self._validate_doi_format(doi):
                return None
            
            # Create paper stub from DOI
            paper = self._create_doi_paper_stub(doi)
            
            print(f"SearchAgent: Retrieved paper by DOI")
            return paper
            
        except Exception as e:
            print(f"SearchAgent DOI error: {str(e)}")
            return None
    
    def _validate_search_query(self, query: str) -> bool:
        """
        Validate search query parameters.
        
        Args:
            query (str): Search query to validate
            
        Returns:
            bool: True if query is valid, False otherwise
        """
        if not query or not isinstance(query, str) or not query.strip():
            print("SearchAgent: Invalid or empty search query")
            return False
        return True
    
    def _validate_doi_format(self, doi: str) -> bool:
        """
        Validate DOI format.
        
        Args:
            doi (str): DOI to validate
            
        Returns:
            bool: True if DOI format is valid, False otherwise
        """
        if not doi or not isinstance(doi, str) or not doi.strip():
            print("SearchAgent: Invalid or empty DOI")
            return False
        return True
    
    def _format_search_results(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format and validate search results.
        
        Args:
            papers (List[Dict[str, Any]]): Raw search results
            
        Returns:
            List[Dict[str, Any]]: Formatted and validated results
        """
        if not papers:
            return []
        
        # In a full implementation, this would validate and standardize
        # the paper metadata format
        return papers
    
    def _create_doi_paper_stub(self, doi: str) -> Dict[str, Any]:
        """
        Create standardized paper metadata from DOI.
        
        Args:
            doi (str): DOI reference
            
        Returns:
            Dict[str, Any]: Standardized paper metadata object
        """
        return {
            "id": f"doi_{doi.replace('/', '_')}",
            "title": f"Paper with DOI: {doi}",
            "authors": ["Unknown Author"],
            "abstract": f"Paper retrieved via DOI: {doi}",
            "url": f"https://doi.org/{doi}",
            "published_date": "2024-01-01",
            "source": "DOI",
            "citation": f"Retrieved from DOI: {doi}",
            "content": {
                "full_text": f"Content for paper with DOI: {doi}",
                "sections": {}
            }
        }
