"""
Citation Management Agent

This module provides citation management capabilities for research papers,
handling citation tracking, formatting in multiple academic styles (APA, MLA, Chicago),
and bibliography generation for research content.

Classes:
    CitationManager: Main agent class for citation operations

Features:
    - Multiple citation formats (APA, MLA, Chicago)
    - Citation database management and tracking
    - Bibliography generation and formatting
    - Content-to-source linking and traceability
    - Author name formatting and date handling
    - URL and arXiv ID processing

Dependencies:
    - typing: Type hints and annotations
    - datetime: Date handling for citations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class CitationManager:
    """
    Citation Management Agent
    
    Agent for managing research paper citations with support for multiple
    academic formatting styles. Handles citation creation, formatting, 
    bibliography generation, and source tracking.
    
    This agent provides:
    - Citation database management and storage
    - Multiple citation format support (APA, MLA, Chicago)
    - Bibliography generation with proper sorting
    - Author name formatting and date extraction
    - Source linking and content traceability
    - Error handling with graceful degradation
    
    Attributes:
        citations (dict): Database of stored citation objects
        
    Methods:
        add_citation(paper): Add paper to citation database
        format_citation(paper, style): Format citation in specified style
        generate_bibliography(papers, style): Create formatted bibliography
        link_content_to_source(content_id, paper_id): Link content to sources
        get_citation_by_id(citation_id): Retrieve citation by ID
        get_all_citations(): Get all stored citations
        
    Private Methods:
        _format_apa_citation(paper): APA style formatting
        _format_mla_citation(paper): MLA style formatting
        _format_chicago_citation(paper): Chicago style formatting
        _create_citation_object(paper): Create citation object
        _extract_year_from_date(date): Extract year from date string
        _format_authors_for_style(authors, style): Format authors by style
        _extract_arxiv_id(paper): Extract arXiv ID from paper data
        _validate_paper_data(paper): Validate paper metadata
    """
    
    def __init__(self):
        """
        Initialize the citation management agent.
        
        Sets up the citation database for storing and managing
        paper citations across multiple academic formats.
        """
        self.citations = {}
    
    def add_citation(self, paper: Dict[str, Any]) -> str:
        """
        Add a paper to the citation database with validation.
        
        Creates a citation object with multiple format styles and stores it
        in the database for future reference and bibliography generation.
        
        Args:
            paper (Dict[str, Any]): Paper object with metadata including title,
                                   authors, publication date, and source information
            
        Returns:
            str: Unique citation ID for referencing the stored citation
            
        Raises:
            No exceptions - all errors are handled gracefully with fallbacks
        """
        try:
            # Validate paper data
            if not self._validate_paper_data(paper):
                return "cite_invalid"
            
            paper_id = paper.get("id", "unknown")
            citation_id = f"cite_{paper_id}"
            
            # Store the citation with all formats
            self.citations[citation_id] = self._create_citation_object(paper)
            
            print(f"CitationManager: Added citation for '{paper.get('title', 'Unknown')}'")
            return citation_id
            
        except Exception as e:
            print(f"CitationManager error adding citation: {str(e)}")
            return "cite_error"
    
    def format_citation(self, paper: Dict[str, Any], style: str = "apa") -> str:
        """
        Format a citation in the specified academic style.
        
        Converts paper metadata into properly formatted citation string
        according to academic standards (APA, MLA, or Chicago).
        
        Args:
            paper (Dict[str, Any]): Paper object with metadata
            style (str): Citation style - 'apa', 'mla', or 'chicago' (default: 'apa')
            
        Returns:
            str: Formatted citation string according to specified style
            
        Raises:
            No exceptions - all errors are handled gracefully with fallbacks
        """
        try:
            # Validate style parameter
            normalized_style = style.lower()
            if normalized_style not in ["apa", "mla", "chicago"]:
                normalized_style = "apa"  # Default fallback
            
            # Route to appropriate formatter
            if normalized_style == "apa":
                return self._format_apa_citation(paper)
            elif normalized_style == "mla":
                return self._format_mla_citation(paper)
            elif normalized_style == "chicago":
                return self._format_chicago_citation(paper)
                
        except Exception as e:
            print(f"CitationManager error formatting citation: {str(e)}")
            return f"Citation error for paper: {paper.get('title', 'Unknown')}"
    
    def _format_apa_citation(self, paper: Dict[str, Any]) -> str:
        """
        Format citation in APA style with proper author and date handling.
        
        Args:
            paper (Dict[str, Any]): Paper object with metadata
            
        Returns:
            str: APA formatted citation string
        """
        try:
            title = paper.get("title", "Unknown Title")
            authors = paper.get("authors", ["Unknown Author"])
            published_date = paper.get("published_date", "n.d.")
            source = paper.get("source", "Unknown Source")
            url = paper.get("url", "")
            
            # Format authors for APA style
            authors_str = self._format_authors_for_style(authors, "apa")
            
            # Extract year from date
            year = self._extract_year_from_date(published_date)
            
            # Build base citation
            citation = f"{authors_str} ({year}). {title}."
            
            # Add source-specific information
            citation += self._add_source_info_apa(paper, source, url)
            
            return citation
            
        except Exception as e:
            print(f"Error formatting APA citation: {str(e)}")
            return f"APA citation error for: {paper.get('title', 'Unknown')}"
    
    def _format_mla_citation(self, paper: Dict[str, Any]) -> str:
        """
        Format citation in MLA style with proper name inversion.
        
        Args:
            paper (Dict[str, Any]): Paper object with metadata
            
        Returns:
            str: MLA formatted citation string
        """
        try:
            title = paper.get("title", "Unknown Title")
            authors = paper.get("authors", ["Unknown Author"])
            published_date = paper.get("published_date", "")
            url = paper.get("url", "")
            
            # Format authors for MLA style
            authors_str = self._format_authors_for_style(authors, "mla")
            
            # Build citation with quoted title
            citation = f'{authors_str} "{title}."'
            
            # Add date and access information
            citation += self._add_date_and_access_info_mla(published_date, url)
            
            return citation
            
        except Exception as e:
            print(f"Error formatting MLA citation: {str(e)}")
            return f"MLA citation error for: {paper.get('title', 'Unknown')}"
    
    def _format_chicago_citation(self, paper: Dict[str, Any]) -> str:
        """
        Format citation in Chicago style with proper punctuation.
        
        Args:
            paper (Dict[str, Any]): Paper object with metadata
            
        Returns:
            str: Chicago formatted citation string
        """
        try:
            title = paper.get("title", "Unknown Title")
            authors = paper.get("authors", ["Unknown Author"])
            published_date = paper.get("published_date", "")
            url = paper.get("url", "")
            
            # Format authors for Chicago style
            authors_str = self._format_authors_for_style(authors, "chicago")
            
            # Extract year
            year = self._extract_year_from_date(published_date)
            
            # Build citation with quoted title
            citation = f'{authors_str}. "{title}."'
            
            # Add year and access information
            if year and year != "n.d.":
                citation += f" {year}."
            
            if url:
                access_date = datetime.now().strftime('%B %d, %Y')
                citation += f" Accessed {access_date}. {url}."
            
            return citation
            
        except Exception as e:
            print(f"Error formatting Chicago citation: {str(e)}")
            return f"Chicago citation error for: {paper.get('title', 'Unknown')}"
    
    def _create_citation_object(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a citation object with all format styles.
        
        Args:
            paper (Dict[str, Any]): Paper object with metadata
            
        Returns:
            Dict[str, Any]: Citation object with multiple formats
        """
        try:
            return {
                "paper_id": paper.get("id", "unknown"),
                "title": paper.get("title", "Unknown Title"),
                "authors": paper.get("authors", ["Unknown Author"]),
                "published_date": paper.get("published_date", "Unknown"),
                "source": paper.get("source", "Unknown"),
                "url": paper.get("url", ""),
                "styles": {
                    "apa": self._format_apa_citation(paper),
                    "mla": self._format_mla_citation(paper),
                    "chicago": self._format_chicago_citation(paper)
                },
                "created_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error creating citation object: {str(e)}")
            return {
                "paper_id": paper.get("id", "unknown"),
                "error": str(e)
            }
    
    def generate_bibliography(self, papers: List[Dict[str, Any]], style: str = "apa") -> str:
        """
        Generate a formatted bibliography for a list of papers.
        
        Creates a complete bibliography with proper sorting and formatting
        according to the specified academic style.
        
        Args:
            papers (List[Dict[str, Any]]): List of paper objects with metadata
            style (str): Citation style for formatting (default: 'apa')
            
        Returns:
            str: Complete formatted bibliography with header and citations
            
        Raises:
            No exceptions - all errors are handled gracefully with fallbacks
        """
        try:
            print(f"CitationManager: Generating bibliography in {style.upper()} style")
            
            # Generate citations for all papers
            citations = []
            for paper in papers:
                if self._validate_paper_data(paper):
                    citation = self.format_citation(paper, style)
                    citations.append(citation)
            
            # Sort citations appropriately
            citations = self._sort_citations_by_style(citations, style)
            
            # Format bibliography with header
            bibliography = self._create_bibliography_header(style)
            for citation in citations:
                bibliography += f"{citation}\n\n"
            
            return bibliography
            
        except Exception as e:
            print(f"CitationManager error generating bibliography: {str(e)}")
            return f"Error generating bibliography: {str(e)}"
    
    def link_content_to_source(self, content_id: str, paper_id: str) -> None:
        """Link content (summary/synthesis) to its source papers.
        
        Args:
            content_id (str): ID of the content (summary/synthesis)
            paper_id (str): ID of the source paper
        """
        try:
            # This could be extended to maintain a more sophisticated
            # content-to-source mapping for detailed traceability
            print(f"CitationManager: Linked content {content_id} to source {paper_id}")
        except Exception as e:
            print(f"CitationManager error linking content: {str(e)}")
    
    def get_citation_by_id(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get citation by ID from the database.
        
        Args:
            citation_id (str): Unique citation identifier
            
        Returns:
            Optional[Dict[str, Any]]: Citation object or None if not found
        """
        return self.citations.get(citation_id, None)
    
    def get_all_citations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all stored citations from the database.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of all citations with IDs as keys
        """
        return self.citations.copy()
    
    def _validate_paper_data(self, paper: Dict[str, Any]) -> bool:
        """
        Validate paper metadata for citation generation.
        
        Args:
            paper (Dict[str, Any]): Paper object to validate
            
        Returns:
            bool: True if paper data is valid for citation, False otherwise
        """
        if not isinstance(paper, dict):
            return False
            
        # Check for required fields
        required_fields = ["title", "authors"]
        for field in required_fields:
            if not paper.get(field):
                return False
                
        # Validate authors field
        authors = paper.get("authors")
        if not isinstance(authors, (list, str)) or len(str(authors).strip()) == 0:
            return False
            
        return True
    
    def _extract_year_from_date(self, date_str: str) -> str:
        """
        Extract year from publication date string.
        
        Args:
            date_str (str): Date string in various formats
            
        Returns:
            str: Extracted year or 'n.d.' if not found
        """
        try:
            if not date_str or date_str in ["Unknown", "n.d.", ""]:
                return "n.d."
                
            # Try to extract first 4 digits as year
            if len(date_str) >= 4 and date_str[:4].isdigit():
                year = int(date_str[:4])
                # Validate reasonable year range
                if 1900 <= year <= 2030:
                    return str(year)
                    
            return "n.d."
            
        except (ValueError, TypeError):
            return "n.d."
    
    def _format_authors_for_style(self, authors: List[str], style: str) -> str:
        """
        Format author names according to citation style requirements.
        
        Args:
            authors (List[str]): List of author names
            style (str): Citation style ('apa', 'mla', 'chicago')
            
        Returns:
            str: Formatted author string according to style
        """
        try:
            if not isinstance(authors, list):
                authors = [str(authors)]
                
            if not authors or len(authors) == 0:
                return "Unknown Author"
            
            if style == "apa":
                return self._format_authors_apa(authors)
            elif style == "mla":
                return self._format_authors_mla(authors)
            elif style == "chicago":
                return self._format_authors_chicago(authors)
            else:
                return self._format_authors_apa(authors)  # Default to APA
                
        except Exception as e:
            print(f"Error formatting authors: {str(e)}")
            return "Unknown Author"
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """
        Format authors for APA style.
        
        Args:
            authors (List[str]): List of author names
            
        Returns:
            str: APA formatted author string
        """
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        else:
            return f"{authors[0]} et al."
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """
        Format authors for MLA style with name inversion.
        
        Args:
            authors (List[str]): List of author names
            
        Returns:
            str: MLA formatted author string
        """
        if len(authors) == 1:
            # Last, First format for MLA
            name_parts = authors[0].split()
            if len(name_parts) >= 2:
                return f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
            else:
                return authors[0]
        else:
            # Multiple authors - invert first author only
            first_author = authors[0].split()
            if len(first_author) >= 2:
                return f"{first_author[-1]}, {' '.join(first_author[:-1])}, et al."
            else:
                return f"{authors[0]}, et al."
    
    def _format_authors_chicago(self, authors: List[str]) -> str:
        """
        Format authors for Chicago style.
        
        Args:
            authors (List[str]): List of author names
            
        Returns:
            str: Chicago formatted author string
        """
        return " and ".join(authors)
    
    def _add_source_info_apa(self, paper: Dict[str, Any], source: str, url: str) -> str:
        """
        Add source-specific information for APA citations.
        
        Args:
            paper (Dict[str, Any]): Paper object
            source (str): Source type
            url (str): Source URL
            
        Returns:
            str: Source information string to append to citation
        """
        try:
            if source == "arXiv":
                arxiv_id = self._extract_arxiv_id(paper)
                if arxiv_id:
                    return f" arXiv:{arxiv_id}"
                else:
                    return f" Retrieved from {url}" if url else ""
                    
            elif source == "DOI":
                return f" {url}" if url else ""
            else:
                return f" Retrieved from {url}" if url else ""
                
        except Exception:
            return f" Retrieved from {url}" if url else ""
    
    def _extract_arxiv_id(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Extract arXiv ID from paper metadata or URL.
        
        Args:
            paper (Dict[str, Any]): Paper object
            
        Returns:
            Optional[str]: arXiv ID if found, None otherwise
        """
        try:
            # Check metadata first
            if "metadata" in paper and "arxiv_id" in paper["metadata"]:
                return paper["metadata"]["arxiv_id"]
            
            # Try to extract from URL
            url = paper.get("url", "")
            if url and "arxiv.org" in url:
                arxiv_id = url.split("/")[-1].replace(".pdf", "")
                return arxiv_id
                
            return None
            
        except Exception:
            return None
    
    def _add_date_and_access_info_mla(self, published_date: str, url: str) -> str:
        """
        Add date and access information for MLA citations.
        
        Args:
            published_date (str): Publication date
            url (str): Source URL
            
        Returns:
            str: Date and access information string
        """
        info = ""
        
        if published_date and published_date != "Unknown":
            info += f" {published_date}."
        
        if url:
            access_date = datetime.now().strftime('%d %b %Y')
            info += f" Web. {access_date}. {url}"
            
        return info
    
    def _sort_citations_by_style(self, citations: List[str], style: str) -> List[str]:
        """
        Sort citations according to style requirements.
        
        Args:
            citations (List[str]): List of formatted citations
            style (str): Citation style
            
        Returns:
            List[str]: Sorted citations
        """
        # Sort alphabetically by first author for APA and MLA
        if style.lower() in ["apa", "mla"]:
            return sorted(citations)
        else:
            # Chicago may use different sorting, but default to alphabetical
            return sorted(citations)
    
    def _create_bibliography_header(self, style: str) -> str:
        """
        Create bibliography header according to style.
        
        Args:
            style (str): Citation style
            
        Returns:
            str: Formatted header string
        """
        style_upper = style.upper()
        if style.lower() == "apa":
            return f"References ({style_upper} Style)\n\n"
        elif style.lower() == "mla":
            return f"Works Cited ({style_upper} Style)\n\n"
        elif style.lower() == "chicago":
            return f"Bibliography ({style_upper} Style)\n\n"
        else:
            return f"References ({style_upper} Style)\n\n"
