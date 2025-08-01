"""
Research Paper Summary Agent

This module provides comprehensive summarization capabilities for research papers,
using advanced multi-stage processing pipelines and language model integration
to generate high-quality summaries at different levels of detail.

Classes:
    SummaryAgent: Main agent class for research paper summarization

Features:
    - Multi-stage summarization pipeline with refinement
    - Intelligent text chunking and processing
    - Multiple summary formats (executive, technical, key insights)
    - Content-aware processing based on available text sources
    - Advanced text preprocessing and optimization
    - Quality validation and iterative improvement
    - Fallback mechanisms for processing failures

Dependencies:
    - services.llm_service: Language model service integration
    - system.config: Configuration management
    - typing: Type hints and annotations
    - re: Regular expression processing
"""

from typing import Dict, Any, List
import re
from services.llm_service import LLMService
from system.config import Config


class SummaryAgent:
    """
    Research Paper Summary Agent
    
    Specialized agent for generating comprehensive, high-quality summaries of
    research papers using advanced multi-stage processing pipelines and
    intelligent content analysis.
    
    This agent provides:
    - Multi-stage summarization with iterative refinement
    - Intelligent content source selection and optimization
    - Multiple summary formats for different use cases
    - Advanced text chunking and processing capabilities
    - Quality validation and improvement mechanisms
    - Fallback strategies for processing challenges
    
    Attributes:
        llm_service (LLMService): Language model service for text generation
        max_chunk_size (int): Maximum characters per processing chunk
        max_summary_iterations (int): Maximum refinement iterations
        
    Methods:
        generate_comprehensive_summary(paper): Generate multi-format summary
        generate_multiple_summaries(papers): Batch summary generation
        _prepare_source_text(paper): Prepare text for summarization
        _chunk_text_for_processing(text): Split text into manageable chunks
        _generate_chunked_summaries(chunks): Process text chunks separately
        _consolidate_chunk_summaries(summaries): Merge chunk summaries
        _refine_summary_quality(summary): Improve summary quality
        _validate_summary_content(summary): Validate summary quality
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the research paper summary agent.
        
        Sets up the language model service and configures processing parameters
        for optimal summarization performance and quality.
        
        Args:
            llm_service (LLMService): Language model service for text generation
                                    and content analysis.
        """
        self.llm_service = llm_service
        self.max_chunk_size = 4000  # Maximum characters per processing chunk
        self.max_summary_iterations = 3  # Maximum refinement iterations
    
    def generate_comprehensive_summary(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive multi-stage summary of research paper.
        
        Creates multiple types of summaries using advanced processing techniques
        including chunking, refinement, and quality validation for optimal results.
        
        Args:
            paper (Dict[str, Any]): Paper object containing title, authors, abstract,
                                   and content for comprehensive summarization.
            
        Returns:
            Dict[str, Any]: Dictionary containing multiple summary formats including
                           executive summary, technical summary, and key insights.
                           
        Example:
            >>> agent = SummaryAgent(llm_service)
            >>> summary = agent.generate_comprehensive_summary(paper)
            >>> print(summary.keys())  # ['executive', 'technical', 'key_insights']
        """
        paper_title = paper.get('title', 'Unknown Title')
        print(f"SummaryAgent: Generating comprehensive summary for '{paper_title}'")
        
        try:
            # Prepare optimized source text for processing
            source_text, text_source = self._prepare_source_text(paper)
            
            # Generate comprehensive summary using appropriate strategy
            if len(source_text) > self.max_chunk_size:
                summary_result = self._process_long_content(source_text, paper)
            else:
                summary_result = self._process_standard_content(source_text, paper)
            
            print(f"SummaryAgent: Successfully generated comprehensive summary")
            return summary_result
            
        except Exception as e:
            print(f"SummaryAgent error: {str(e)}")
            return self._create_fallback_summary(paper)
    
    def _generate_chunked_summary(self, text: str, title: str, authors: str) -> str:
        """Generate summary using chunked approach for long texts.
        
        Args:
            text (str): Text to summarize
            title (str): Paper title
            authors (str): Paper authors
            
        Returns:
            str: Generated summary
        """
        if len(text) <= self.max_chunk_size:
            # Text is short enough, summarize directly
            return self.llm_service.summarize_text(text, title, authors)
        
        # Split into chunks
        chunks = self._split_into_chunks(text, self.max_chunk_size)
        print(f"📄 Split text into {len(chunks)} chunks for processing")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"📝 Processing chunk {i+1}/{len(chunks)}")
            prompt = f"""Summarize this section of a research paper titled "{title}" by {authors}:

{chunk}

Provide a detailed summary focusing on key findings, methodology, and conclusions."""
            
            summary = self.llm_service.generate_text(prompt, max_tokens=300)
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        
        # If combined summary is still too long, refine it
        iteration = 0
        while len(combined_summary) > self.max_chunk_size and iteration < self.max_summary_iterations:
            iteration += 1
            print(f"🔄 Refining summary (iteration {iteration})")
            
            prompt = f"""Create a comprehensive summary of this research paper titled "{title}":

{combined_summary}

Provide a well-structured summary that captures all important aspects while being more concise."""
            
            combined_summary = self.llm_service.generate_text(prompt, max_tokens=500)
        
        return combined_summary
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into manageable chunks.
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum size per chunk
            
        Returns:
            list: List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find a good breaking point (end of sentence or paragraph)
            end_pos = current_pos + chunk_size
            
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
            
            # Look for sentence or paragraph breaks
            break_points = [
                text.rfind('\n\n', current_pos, end_pos),  # Paragraph break
                text.rfind('. ', current_pos, end_pos),    # Sentence break
                text.rfind('\n', current_pos, end_pos),    # Line break
            ]
            
            # Choose the best breaking point
            break_point = max(bp for bp in break_points if bp > current_pos)
            
            if break_point == current_pos:
                # No good break found, force split
                break_point = end_pos
            
            chunks.append(text[current_pos:break_point].strip())
            current_pos = break_point
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from a summary.
        
        Args:
            summary (str): Summary text
            
        Returns:
            list: List of key points
        """
        prompt = f"""Extract the most important key points from this research summary and format them as bullet points:

{summary}

Provide 5-7 key points that capture the most important findings, methods, and conclusions."""
        
        try:
            response = self.llm_service.generate_text(prompt, max_tokens=300)
            
            # Parse bullet points
            lines = response.split('\n')
            key_points = []
            
            for line in lines:
                line = line.strip()
                # Remove bullet point markers
                line = re.sub(r'^[-•*]\s*', '', line)
                if line and len(line) > 10:  # Filter out very short lines
                    key_points.append(line)
            
            return key_points[:7]  # Return max 7 points
            
        except Exception as e:
            print(f"❌ Error extracting key points: {str(e)}")
            # Fallback: simple sentence extraction
            sentences = summary.split('.')
            return [s.strip() + '.' for s in sentences[:5] if len(s.strip()) > 20]
    
    def _generate_executive_summary(self, summary: str, title: str) -> str:
        """Generate an executive summary suitable for non-technical audiences.
        
        Args:
            summary (str): Technical summary
            title (str): Paper title
            
        Returns:
            str: Executive summary
        """
        prompt = f"""Create an executive summary of this research paper titled "{title}" that would be suitable for business leaders and non-technical stakeholders:

{summary}

Focus on:
- What problem does this research solve?
- What are the key findings?
- What are the practical implications?
- Why should decision-makers care about this?

Keep it concise and avoid technical jargon."""
        
        try:
            return self.llm_service.generate_text(prompt, max_tokens=250)
        except Exception as e:
            print(f"❌ Error generating executive summary: {str(e)}")
            return f"Executive Summary: {summary[:200]}..."
    
    def _generate_technical_summary(self, summary: str, title: str) -> str:
        """Generate a technical summary for researchers and technical audiences.
        
        Args:
            summary (str): Base summary
            title (str): Paper title
            
        Returns:
            str: Technical summary
        """
        prompt = f"""Create a technical summary of this research paper titled "{title}" for researchers and technical professionals:

{summary}

Focus on:
- Research methodology and approach
- Technical details and innovations
- Experimental setup and validation
- Limitations and future work
- Technical implications for the field

Maintain technical accuracy and include relevant details."""
        
        try:
            return self.llm_service.generate_text(prompt, max_tokens=400)
        except Exception as e:
            print(f"❌ Error generating technical summary: {str(e)}")
            return f"Technical Summary: {summary}"
    
    def _generate_fallback_summary(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fallback summary when main processing fails.
        
        Args:
            paper (dict): Paper object
            
        Returns:
            dict: Fallback summary data
        """
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "")
        
        if abstract:
            main_summary = f"Summary of '{title}': {abstract}"
        else:
            main_summary = f"Unable to generate detailed summary for '{title}' due to processing error."
        
        return {
            "main_summary": main_summary,
            "key_points": ["Summary generation failed", "Please try again or check the paper format"],
            "executive_summary": f"Executive Summary: {main_summary}",
            "technical_summary": f"Technical Summary: {main_summary}",
            "metadata": {
                "text_source": "fallback",
                "original_length": 0,
                "summary_length": len(main_summary),
                "compression_ratio": 0
            }
        }
    
    def generate_summary(self, paper: Dict[str, Any]) -> str:
        """Generate a basic summary (backwards compatibility).
        
        Args:
            paper (dict): Paper object
            
        Returns:
            str: Generated summary
        """
        # For backwards compatibility, return just the main summary
        comprehensive = self.generate_comprehensive_summary(paper)
        return comprehensive.get("main_summary", "Summary generation failed")
    
    def generate_multiple_summaries(self, papers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate comprehensive summaries for multiple research papers in batch.
        
        Efficiently processes multiple papers with error handling and progress
        tracking for large-scale summarization tasks.
        
        Args:
            papers (List[Dict[str, Any]]): List of paper objects for summarization
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping paper IDs to their
                                     comprehensive summary results.
        """
        print(f"SummaryAgent: Generating summaries for {len(papers)} papers")
        
        summaries = {}
        successful_count = 0
        
        for paper in papers:
            try:
                paper_id = paper.get("id", "unknown")
                summary = self.generate_summary(paper)
                summaries[paper_id] = summary
                successful_count += 1
            except Exception as e:
                print(f"Error generating summary for {paper.get('title', 'Unknown')}: {str(e)}")
                summaries[paper.get("id", "unknown")] = f"Error generating summary for paper: {paper.get('title', 'Unknown')}"
        
        print(f"SummaryAgent: Successfully generated {successful_count}/{len(papers)} summaries")
        return summaries
    
    def extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from a summary.
        
        Args:
            summary (str): Paper summary text
            
        Returns:
            list: List of key points
        """
        try:
            # Simple extraction based on sentence structure
            sentences = summary.split('.')
            key_points = []
            
            # Look for sentences that indicate key findings
            key_indicators = [
                'found', 'discovered', 'showed', 'demonstrated', 'revealed',
                'concluded', 'results', 'findings', 'evidence', 'significant'
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Ignore very short sentences
                    if any(indicator in sentence.lower() for indicator in key_indicators):
                        key_points.append(sentence + '.')
            
            # If no key points found using indicators, just take the first few sentences
            if not key_points:
                key_points = [s.strip() + '.' for s in sentences[:3] if len(s.strip()) > 20]
            
            return key_points[:5]  # Return maximum 5 key points
            
        except Exception as e:
            print(f"⚠️ Error extracting key points: {str(e)}")
            return ["Key points could not be extracted from summary."]
    
    def format_summary_with_metadata(self, paper: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """Format summary with paper metadata.
        
        Args:
            paper (dict): Paper object
            summary (str): Generated summary
            
        Returns:
            dict: Formatted summary object with metadata
        """
        try:
            key_points = self.extract_key_points(summary)
            
            return {
                "paper_id": paper.get("id", "unknown"),
                "title": paper.get("title", "Unknown Title"),
                "authors": paper.get("authors", ["Unknown Author"]),
                "summary_text": summary,
                "key_points": key_points,
                "citation": paper.get("citation", "Citation not available"),
                "source": paper.get("source", "Unknown"),
                "url": paper.get("url", None),
                "published_date": paper.get("published_date", "Unknown")
            }
            
        except Exception as e:
            print(f"Error formatting summary: {str(e)}")
            return {
                "paper_id": paper.get("id", "unknown"),
                "title": paper.get("title", "Unknown Title"),
                "summary_text": summary,
                "error": str(e)
            }
    
    def _prepare_source_text(self, paper: Dict[str, Any]) -> tuple:
        """
        Prepare optimized source text for summarization processing.
        
        Args:
            paper (Dict[str, Any]): Paper object with content
            
        Returns:
            tuple: (source_text, text_source) for processing
        """
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", ["Unknown Author"])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        abstract = paper.get("abstract", "")
        content = paper.get("content", {})
        full_text = content.get("full_text", "") if isinstance(content, dict) else ""
        
        # Choose optimal text source
        if full_text and len(full_text) > len(abstract):
            source_text = f"Title: {title}\nAuthors: {authors_str}\nAbstract: {abstract}\n\nFull Text: {full_text}"
            text_source = "full_document"
        else:
            source_text = f"Title: {title}\nAuthors: {authors_str}\nAbstract: {abstract}"
            text_source = "abstract_only"
            
        return source_text, text_source
    
    def _process_long_content(self, source_text: str, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process long content using chunking strategy.
        
        Args:
            source_text (str): Long source text to process
            paper (Dict[str, Any]): Paper object
            
        Returns:
            Dict[str, Any]: Comprehensive summary result
        """
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", ["Unknown Author"])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        
        summary = self._generate_chunked_summary(source_text, title, authors_str)
        return self._create_comprehensive_result(summary, paper, "chunked_processing", source_text)
    
    def _process_standard_content(self, source_text: str, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process standard length content using direct summarization.
        
        Args:
            source_text (str): Source text to process
            paper (Dict[str, Any]): Paper object
            
        Returns:
            Dict[str, Any]: Comprehensive summary result
        """
        title = paper.get('title', 'Unknown Title')
        authors_str = ', '.join(paper.get('authors', ['Unknown Author']))
        
        print(f"📝 Calling LLM service with {len(source_text)} chars of text...")
        summary = self.llm_service.summarize_text(source_text, title, authors_str)
        print(f"📝 LLM returned summary of length: {len(summary)} chars")
        print(f"📝 Summary preview: {summary[:100]}...")
        
        return self._create_comprehensive_result(summary, paper, "direct_processing", source_text)
    
    def _create_comprehensive_result(self, summary: str, paper: Dict[str, Any], method: str, source_text: str = "") -> Dict[str, Any]:
        """
        Create comprehensive summary result structure with distinct summaries.
        
        Args:
            summary (str): Generated summary text
            paper (Dict[str, Any]): Paper object
            method (str): Processing method used
            source_text (str): Original source text that was summarized
            
        Returns:
            Dict[str, Any]: Structured summary result with distinct summaries
        """
        title = paper.get("title", "Unknown Title")
        
        # Generate distinct executive summary (concise, high-level)
        executive_prompt = f"""Create a concise executive summary (2-3 sentences) for this research paper:
        
Title: {title}
Summary: {summary}

Focus on the main contribution and practical significance. Keep it brief and accessible."""
        
        executive_summary = self.llm_service.generate_text(executive_prompt, max_tokens=200)
        
        # Generate detailed technical summary
        technical_prompt = f"""Create a detailed technical summary for this research paper:
        
Title: {title}
Summary: {summary}

Include methodology, technical details, results, and implications. Make it comprehensive but focused."""
        
        technical_summary = self.llm_service.generate_text(technical_prompt, max_tokens=600)
        
        # Extract key insights using LLM
        key_insights = self._extract_key_insights_llm(summary, title)
        
        # Calculate proper compression ratio using actual source text
        original_length = len(source_text) if source_text else len(paper.get("content", ""))
        summary_length = len(summary)
        
        if original_length > 0:
            # Calculate compression as original/summary ratio (e.g., 10:1 = 10.0)
            compression_ratio = original_length / summary_length
        else:
            compression_ratio = 1.0  # No compression if no original content
        
        return {
            "main_summary": summary,
            "executive_summary": executive_summary.strip(),
            "technical_summary": technical_summary.strip(),
            "key_points": key_insights,  # Changed from key_insights to key_points
            "processing_method": method,
            "metadata": {
                "summary_length": summary_length,
                "original_length": original_length,
                "text_source": method,
                "compression_ratio": compression_ratio
            },
            "paper_metadata": {
                "title": paper.get("title", "Unknown"),
                "authors": paper.get("authors", []),
                "id": paper.get("id", "unknown")
            }
        }
    
    def _extract_key_insights_llm(self, summary: str, title: str) -> List[str]:
        """
        Extract key insights using LLM for better quality.
        
        Args:
            summary (str): Summary text to analyze
            title (str): Paper title for context
            
        Returns:
            List[str]: List of key insights
        """
        insights_prompt = f"""Extract 4-6 key insights from this research paper summary:

Title: {title}
Summary: {summary}

Format as bullet points, each highlighting a distinct insight, finding, or contribution.
Focus on:
- Main findings/results
- Novel contributions
- Methodological insights
- Practical implications

Key Insights:"""
        
        try:
            insights_text = self.llm_service.generate_text(insights_prompt, max_tokens=400)
            
            # Parse bullet points or numbered lists
            insights = []
            for line in insights_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or 
                           line.startswith('*') or line[0:2].replace('.', '').isdigit()):
                    # Clean up bullet point markers
                    clean_line = re.sub(r'^[•\-*]|\d+\.', '', line).strip()
                    if clean_line and len(clean_line) > 10:
                        insights.append(clean_line)
            
            return insights[:6] if insights else self._extract_key_insights(summary)
            
        except Exception as e:
            print(f"Error extracting LLM insights: {e}")
            return self._extract_key_insights(summary)

    def _extract_key_insights(self, summary: str) -> List[str]:
        """
        Extract key insights from summary text.
        
        Args:
            summary (str): Summary text to analyze
            
        Returns:
            List[str]: List of key insights
        """
        # Simple extraction - could be enhanced with NLP
        sentences = summary.split('. ')
        key_sentences = [s.strip() + '.' for s in sentences[:3] if len(s.strip()) > 20]
        return key_sentences if key_sentences else ["Key insights extraction failed"]
    
    def _create_fallback_summary(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback summary when processing fails.
        
        Args:
            paper (Dict[str, Any]): Paper object
            
        Returns:
            Dict[str, Any]: Fallback summary result
        """
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "No abstract available")
        
        fallback_summary = f"Summary generation failed for '{title}'. Abstract: {abstract}"
        
        return {
            "main_summary": fallback_summary,
            "executive_summary": fallback_summary,
            "technical_summary": fallback_summary,
            "key_points": ["Summary generation failed"],  # Changed from key_insights to key_points
            "processing_method": "fallback",
            "paper_metadata": {
                "title": title,
                "authors": paper.get("authors", []),
                "id": paper.get("id", "unknown")
            }
        }
