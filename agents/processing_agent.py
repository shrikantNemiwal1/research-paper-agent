"""
Document Processing Agent

This module handles the extraction and processing of research papers from various sources
including PDF files, web URLs, and DOI references. It provides unified content extraction
with metadata parsing and text cleaning capabilities.

Classes:
    ProcessingAgent: Main class for processing papers from multiple sources

Features:
    - PDF text extraction with metadata
    - URL content processing (PDF and HTML)
    - DOI resolution and content retrieval
    - Text cleaning and normalization
    - Section extraction and structuring

Dependencies:
    - PyPDF2: PDF processing
    - requests: HTTP operations
    - system.config: Configuration management
"""

import os
import requests
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
import PyPDF2
from urllib.parse import urlparse
import re
import time
from system.config import Config


class ProcessingAgent:
    """
    Agent responsible for processing research papers from various sources.
    
    This class provides unified document processing capabilities for research papers
    in different formats and from different sources. It handles content extraction,
    metadata parsing, and text normalization.
    
    Attributes:
        max_text_length: Maximum length of extracted text content
    
    Methods:
        process_pdf: Process uploaded PDF files
        process_url: Process papers from web URLs
        process_doi: Process papers from DOI references
        
    Private Methods:
        _process_pdf_from_bytes: Internal PDF processing from byte content
        _process_html_content: Internal HTML content processing
        _extract_title_and_abstract: Extract key metadata from text
        _extract_basic_sections: Structure text into sections
        _clean_text: Normalize and clean extracted text
    """
    
    def __init__(self):
        """
        Initialize the processing agent with configuration settings.
        """
        self.max_text_length = Config.MAX_TEXT_LENGTH
    
    def process_pdf(self, file) -> Optional[Dict[str, Any]]:
        """
        Process an uploaded PDF file and extract content with metadata.
        
        Extracts text content from all pages of a PDF file, attempts to parse
        metadata from the PDF properties, and structures the content for further
        processing. Handles various PDF formats and encoding issues gracefully.
        
        Args:
            file: Uploaded PDF file object with read() method
            
        Returns:
            Dict containing:
                - id: Unique identifier for the paper
                - title: Extracted or derived title
                - authors: List of author names
                - abstract: Paper abstract or summary
                - content: Full text and structured sections
                - metadata: Source information and processing details
                
            Returns None if processing fails completely.
        """
        print(f"Processing PDF file '{file.name}'")
        
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            
            full_text = ""
            metadata = {}
            
            # Extract PDF metadata if available
            if pdf_reader.metadata:
                metadata.update({
                    'pdf_title': pdf_reader.metadata.get('/Title', ''),
                    'pdf_author': pdf_reader.metadata.get('/Author', ''),
                    'pdf_subject': pdf_reader.metadata.get('/Subject', ''),
                    'pdf_creator': pdf_reader.metadata.get('/Creator', ''),
                })
            
            # Extract text from pages with limit
            for page_num, page in enumerate(pdf_reader.pages):
                if page_num >= Config.PDF_MAX_PAGES:
                    print(f"Reached max page limit ({Config.PDF_MAX_PAGES}), stopping extraction")
                    break
                try:
                    page_text = page.extract_text()
                    full_text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
            
            # Enhanced text cleaning and structure extraction
            cleaned_text, extracted_metadata = self._extract_paper_structure(full_text)
            metadata.update(extracted_metadata)
            
            # Extract title from filename or PDF metadata
            title = self._extract_title(file.name, metadata)
            authors = self._extract_authors(cleaned_text, metadata)
            abstract = self._extract_abstract(cleaned_text)
            
            paper_data = {
                "id": self._generate_paper_id(title),
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "content": {
                    "full_text": cleaned_text,
                    "page_count": len(pdf_reader.pages),
                    "word_count": len(cleaned_text.split())
                },
                "metadata": metadata,
                "source": {
                    "type": "pdf_upload",
                    "filename": file.name,
                    "size_bytes": len(file.read())
                }
            }
            
            print(f"✅ ProcessingAgent: Successfully processed PDF - {len(cleaned_text)} chars, {len(pdf_reader.pages)} pages")
            return paper_data
            
        except Exception as e:
            print(f"Error processing PDF '{file.name}': {str(e)}")
            return None
    
    def _extract_paper_structure(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Extract paper structure and metadata from text.
        
        Args:
            text (str): Raw text from PDF
            
        Returns:
            tuple: (cleaned_text, extracted_metadata)
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Extract metadata
        metadata = {}
        
        # Try to identify paper sections
        sections = self._identify_sections(cleaned_text)
        if sections:
            metadata['sections'] = sections
        
        # Extract references count
        ref_count = len(re.findall(r'\[\d+\]|\(\d{4}\)', cleaned_text))
        metadata['reference_count'] = ref_count
        
        # Estimate reading time (average 200 words per minute)
        word_count = len(cleaned_text.split())
        metadata['estimated_reading_time_minutes'] = max(1, word_count // 200)
        
        return cleaned_text, metadata
    
    def _extract_title(self, filename: str, metadata: Dict[str, Any]) -> str:
        """Extract paper title from filename or metadata.
        
        Args:
            filename (str): PDF filename
            metadata (dict): Extracted metadata
            
        Returns:
            str: Paper title
        """
        # First try PDF metadata
        if metadata.get('pdf_title'):
            return metadata['pdf_title'].strip()
        
        # Clean up filename as fallback
        title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        # Remove common academic paper prefixes
        title = re.sub(r'^\d+[\-\.]?\s*', '', title)  # Remove leading numbers
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    def _extract_authors(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract authors from text or metadata.
        
        Args:
            text (str): Paper text
            metadata (dict): Extracted metadata
            
        Returns:
            list: List of author names
        """
        authors = []
        
        # Try PDF metadata first
        if metadata.get('pdf_author'):
            authors = [metadata['pdf_author'].strip()]
        
        # Try to extract from text (look for common patterns)
        if not authors:
            # Look for author patterns in first few paragraphs
            first_part = text[:2000]
            
            # Pattern: "Author Name1, Author Name2"
            author_patterns = [
                r'(?:Authors?|By):\s*([A-Z][a-zA-Z\s,\.]+?)(?:\n|\r|Abstract|Introduction)',
                r'^([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)*)\s*(?:\n|\r)',
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, first_part, re.MULTILINE | re.IGNORECASE)
                if match:
                    author_text = match.group(1).strip()
                    # Split by comma and clean
                    authors = [name.strip() for name in author_text.split(',') if name.strip()]
                    break
        
        return authors if authors else ["Unknown Author"]
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from paper text.
        
        Args:
            text (str): Paper text
            
        Returns:
            str: Abstract text
        """
        # Look for abstract section
        abstract_patterns = [
            r'Abstract\s*[:\-]?\s*(.*?)(?:\n\s*\n|\n\s*1\.|\n\s*Introduction|\n\s*Keywords)',
            r'ABSTRACT\s*[:\-]?\s*(.*?)(?:\n\s*\n|\n\s*1\.|\n\s*INTRODUCTION|\n\s*Keywords)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Ensure it's not too short
                    return abstract
        
        # Fallback: return first substantial paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 100 and not para.startswith(('Figure', 'Table', 'Fig.')):
                return para[:500] + "..." if len(para) > 500 else para
        
        return ""
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify paper sections from text.
        
        Args:
            text (str): Paper text
            
        Returns:
            list: List of identified sections
        """
        sections = []
        
        # Common section patterns
        section_patterns = [
            r'\n\s*(\d+\.?\s+[A-Z][A-Za-z\s]+)\n',
            r'\n\s*([A-Z]{2,}[A-Z\s]*)\n',
            r'\n\s*(Introduction|Abstract|Methodology|Results|Discussion|Conclusion|References)\s*\n',
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                section_title = match.group(1).strip()
                start_pos = match.start()
                sections.append({
                    'title': section_title,
                    'start_position': start_pos
                })
        
        return sections
    
    def _generate_paper_id(self, title: str) -> str:
        """Generate a unique paper ID from title.
        
        Args:
            title (str): Paper title
            
        Returns:
            str: Generated paper ID
        """
        # Create ID from title
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
        words = clean_title.split()[:5]  # First 5 words
        paper_id = '_'.join(words)
        
        # Add timestamp for uniqueness
        timestamp = str(int(time.time()))[-6:]  # Last 6 digits
        
        return f"upload_{paper_id}_{timestamp}"
    
    def process_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Process a research paper from a web URL.
        
        Downloads content from the provided URL and processes it based on content type.
        Handles both PDF files and HTML pages with appropriate extraction methods.
        
        Args:
            url: Web URL pointing to a research paper or document
            
        Returns:
            Dict containing processed paper data with content and metadata,
            or None if processing fails
        """
        print(f"Processing URL '{url}'")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'application/pdf' in content_type:
                return self._process_pdf_from_bytes(response.content, url)
            else:
                return self._process_html_content(response.text, url)
                
        except Exception as e:
            print(f"URL processing error: {str(e)}")
            return None
    
    def process_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Process a research paper from a DOI reference.
        
        Resolves the DOI to an actual URL and processes the content. Provides
        graceful fallback if the resolved content is not accessible.
        
        Args:
            doi: Digital Object Identifier for the paper
            
        Returns:
            Dict containing processed paper data, or minimal paper object
            if content cannot be fully accessed
        """
        print(f"Processing DOI '{doi}'")
        
        try:
            doi_url = f"https://doi.org/{doi}"
            response = requests.get(doi_url, timeout=30, allow_redirects=True)
            
            if response.status_code == 200:
                return self.process_url(response.url)
            else:
                # Create minimal paper object for inaccessible content
                paper = {
                    "id": f"doi_{doi.replace('/', '_').replace('.', '_')}",
                    "title": f"Paper with DOI: {doi}",
                    "authors": ["Unknown Author"],
                    "abstract": f"Paper retrieved via DOI: {doi}. Full text not available.",
                    "url": doi_url,
                    "published_date": "Unknown",
                    "source": "DOI",
                    "citation": f"DOI: {doi}",
                    "content": {
                        "full_text": f"Paper with DOI: {doi}. Full text could not be retrieved.",
                        "sections": {}
                    }
                }
                
                print("DOI resolved but content not fully accessible")
                return paper
                
        except Exception as e:
            print(f"DOI processing error: {str(e)}")
            return None
    
    def _process_pdf_from_bytes(self, pdf_bytes: bytes, source_url: str) -> Optional[Dict[str, Any]]:
        """Process PDF from byte content."""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            
            # Extract text
            full_text = ""
            for page in pdf_reader.pages:
                try:
                    full_text += page.extract_text() + "\n"
                except:
                    continue
            
            full_text = self._clean_text(full_text)
            
            # Extract metadata
            title, abstract = self._extract_title_and_abstract(full_text)
            if not title:
                title = f"Paper from {urlparse(source_url).hostname}"
            
            paper = {
                "id": f"url_{source_url.split('/')[-1].replace('.pdf', '')}",
                "title": title,
                "authors": ["Unknown Author"],
                "abstract": abstract or full_text[:500] + "..." if len(full_text) > 500 else full_text,
                "url": source_url,
                "published_date": "Unknown",
                "source": "URL",
                "citation": f"Retrieved from {source_url}",
                "content": {
                    "full_text": full_text,
                    "sections": self._extract_basic_sections(full_text)
                }
            }
            
            return paper
            
        except Exception as e:
            print(f"Error processing PDF from bytes: {str(e)}")
            return None
    
    def _process_html_content(self, html_content: str, source_url: str) -> Optional[Dict[str, Any]]:
        """Process HTML content (simplified)."""
        try:
            # For minimal implementation, just extract basic text
            # In a full implementation, you'd use BeautifulSoup for better parsing
            
            # Simple text extraction (remove HTML tags)
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = self._clean_text(text)
            
            # Extract basic metadata
            title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
            title = title_match.group(1) if title_match else f"Content from {urlparse(source_url).hostname}"
            
            paper = {
                "id": f"web_{source_url.split('/')[-1][:20]}",
                "title": title,
                "authors": ["Unknown Author"],
                "abstract": text[:500] + "..." if len(text) > 500 else text,
                "url": source_url,
                "published_date": "Unknown",
                "source": "Web",
                "citation": f"Retrieved from {source_url}",
                "content": {
                    "full_text": text,
                    "sections": {}
                }
            }
            
            return paper
            
        except Exception as e:
            print(f"Error processing HTML content: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', ' ')
        text = text.replace('\uf020', ' ')
        
        return text
    
    def _extract_title_and_abstract(self, text: str) -> tuple:
        """Extract title and abstract from paper text."""
        try:
            lines = text.split('\n')
            title = None
            abstract = None
            
            # Simple heuristic for title (first substantial line)
            for line in lines:
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    title = line
                    break
            
            # Simple heuristic for abstract
            abstract_markers = ['abstract', 'summary', 'introduction']
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(marker in line_lower for marker in abstract_markers):
                    # Try to get the next few lines as abstract
                    abstract_lines = []
                    for j in range(i+1, min(i+10, len(lines))):
                        if lines[j].strip():
                            abstract_lines.append(lines[j].strip())
                        if len(' '.join(abstract_lines)) > 300:
                            break
                    abstract = ' '.join(abstract_lines)
                    break
            
            return title, abstract
            
        except Exception as e:
            print(f"Error extracting title/abstract: {str(e)}")
            return None, None
    
    def _extract_basic_sections(self, text: str) -> Dict[str, str]:
        """Extract basic sections from paper text."""
        try:
            sections = {}
            
            # Simple section detection based on common headers
            section_markers = {
                'introduction': ['introduction', 'intro'],
                'methods': ['methods', 'methodology', 'approach'],
                'results': ['results', 'findings', 'experiments'],
                'conclusion': ['conclusion', 'conclusions', 'summary']
            }
            
            lines = text.split('\n')
            current_section = None
            section_content = []
            
            for line in lines:
                line_lower = line.lower().strip()
                
                # Check if this line is a section header
                section_found = None
                for section_name, markers in section_markers.items():
                    if any(marker in line_lower for marker in markers) and len(line.strip()) < 50:
                        section_found = section_name
                        break
                
                if section_found:
                    # Save previous section
                    if current_section and section_content:
                        sections[current_section] = ' '.join(section_content)
                    
                    # Start new section
                    current_section = section_found
                    section_content = []
                else:
                    # Add to current section
                    if current_section and line.strip():
                        section_content.append(line.strip())
            
            # Save last section
            if current_section and section_content:
                sections[current_section] = ' '.join(section_content)
            
            return sections
            
        except Exception as e:
            print(f"Error extracting sections: {str(e)}")
            return {}
