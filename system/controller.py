"""
Core Research System Controller

This module serves as the central orchestrator for the Research Paper Multi-Agent System.
It coordinates all agent activities and workflows including paper retrieval, processing,
summarization, synthesis, and audio generation.

Classes:
    ResearchSystem: Main controller class that manages the entire research workflow

Dependencies:
    - agents: Search, Processing, Topic Classification, Summary, Synthesis, Podcast, Citation
    - services: LLM Service for AI operations
    - system: Configuration management
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from agents.search_agent import SearchAgent
from agents.processing_agent import ProcessingAgent  
from agents.topic_agent import TopicClassificationAgent
from agents.summary_agent import SummaryAgent
from agents.synthesis_agent import SynthesisAgent
from agents.podcast_agent import PodcastAgent
from agents.citation_agent import CitationManager
from services.llm_service import LLMService
from system.config import Config


class ResearchSystem:
    """
    Central orchestrator for the Research Paper Multi-Agent System.
    
    Coordinates all agents and manages the complete research workflow
    from paper collection to final output generation including summaries and podcasts.
    
    Attributes:
        llm_service: Service for language model operations
        search_agent: Handles paper search operations
        processing_agent: Processes papers from various sources
        topic_agent: Classifies papers by topic
        summary_agent: Generates paper summaries
        synthesis_agent: Creates topic syntheses
        podcast_agent: Generates audio content
        citation_manager: Handles citation formatting
        current_request: State storage for ongoing request
        current_request_id: Unique identifier for current session
    
    Methods:
        process_request: Main entry point for processing research requests
        _reset_state: Clears current request state
        _save_state: Persists current state to file
        load_state: Retrieves saved state from file
    """
    
    def __init__(self, progress_callback=None):
        """
        Initialize the research system with all required agents and services.
        
        Args:
            progress_callback: Optional function to call for progress updates
            
        Raises:
            ValueError: If system configuration is invalid
        """
        # Validate configuration
        if not Config.validate_config():
            raise ValueError("Invalid configuration. Please check environment variables.")
        
        self.progress_callback = progress_callback
        
        # Initialize services
        self.llm_service = LLMService()
        
        # Initialize all agents
        self.search_agent = SearchAgent()
        self.processing_agent = ProcessingAgent()
        self.topic_agent = TopicClassificationAgent(self.llm_service)
        self.summary_agent = SummaryAgent(self.llm_service)
        self.synthesis_agent = SynthesisAgent(self.llm_service)
        self.podcast_agent = PodcastAgent(self.llm_service)
        self.citation_manager = CitationManager()
        
        # Initialize request state storage
        self.current_request = {
            "papers": {},            # paper_id -> paper_object
            "classifications": {},   # paper_id -> topic
            "summaries": {},         # paper_id -> summary_object
            "syntheses": {},         # topic -> synthesis_object
            "audio_files": {},       # content_id -> audio_path
            "citations": {}          # paper_id -> citation_object
        }
        
        self.current_request_id = None
    
    def _log_progress(self, stage: str, message: str, progress: float = None, paper_title: str = None):
        """
        Log progress updates with structured information.
        
        Args:
            stage: Current processing stage
            message: Progress message
            progress: Progress percentage (0.0 to 1.0)
            paper_title: Currently processing paper title
        """
        log_msg = f"[{stage}] {message}"
        if paper_title:
            log_msg += f" | Paper: {paper_title[:50]}..."
        
        print(log_msg)
        
        if self.progress_callback:
            self.progress_callback({
                "stage": stage,
                "message": message,
                "progress": progress,
                "paper_title": paper_title
            })
    
    def process_request(self, query: Optional[str] = None, topics: Optional[List[str]] = None,
                       filters: Optional[Dict] = None, files: Optional[List] = None,
                       urls: Optional[List[str]] = None, dois: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a complete research request through the multi-agent pipeline.
        
        This method orchestrates the entire research workflow:
        1. Collects papers from various sources (arXiv, files, URLs, DOIs)
        2. Classifies papers by user-defined topics
        3. Generates comprehensive summaries for each paper
        4. Creates topic syntheses for papers in the same category
        5. Generates podcast-style audio content
        6. Formats citations for all papers
        
        Args:
            query (str, optional): Research topic/question for arXiv search
            topics (List[str], optional): User-defined topics for classification
            filters (Dict, optional): Search filters (date_range, max_results, etc.)
            files (List, optional): Uploaded PDF files to process
            urls (List[str], optional): URLs pointing to research papers
            dois (List[str], optional): DOI references to resolve and process
            
        Returns:
            Dict[str, Any]: Complete research results containing:
                - papers: List of processed paper objects
                - summaries: List of generated summaries
                - syntheses: List of topic syntheses
                - audio_files: List of generated audio content
                - citations: Dictionary of formatted citations
                - classified_papers: Papers grouped by topic
                - processing_time: Timestamp of completion
                
        Raises:
            Exception: Any error during processing is caught and returned in error field
        """
        try:
            # Initialize request session
            self.current_request_id = str(uuid.uuid4())
            self._reset_state()
            
            # Use default topics if none provided
            if not topics:
                topics = Config.DEFAULT_TOPICS[:3]
            
            self._log_progress("INITIALIZATION", "Starting research pipeline", 0.0)
            
            # Step 1: Collect papers from all sources
            self._log_progress("COLLECTION", "Collecting papers from sources", 0.1)
            papers = self._collect_papers(query, filters, files, urls, dois)
            
            if not papers:
                return self._create_error_response("No papers found. Please check your search query or try different inputs.")
            
            self._log_progress("COLLECTION", f"Successfully collected {len(papers)} papers", 0.2)
            
            # Store papers in state
            for paper in papers:
                self.current_request["papers"][paper["id"]] = paper
            
            # Step 2: Classify papers by topic
            self._log_progress("CLASSIFICATION", "Classifying papers by topic", 0.25)
            # Step 2: Classify papers by topic
            self._log_progress("CLASSIFICATION", "Classifying papers by topic", 0.25)
            classified_papers = self._classify_papers(papers, topics)
            
            # Step 3: Generate comprehensive summaries
            self._log_progress("SUMMARIZATION", "Generating paper summaries", 0.4)
            summaries = self._generate_summaries(papers)
            
            # Step 4: Generate topic syntheses
            self._log_progress("SYNTHESIS", "Creating topic syntheses", 0.65)
            syntheses = self._generate_syntheses(classified_papers)
            
            # Step 5: Generate audio content
            self._log_progress("AUDIO", "Generating audio content", 0.75)
            audio_files = self._generate_audio_content(summaries, syntheses)
            
            # Step 6: Generate citations
            self._log_progress("CITATIONS", "Generating citations", 0.9)
            citations = self._generate_citations(papers)
            
            # Step 7: Save state and return results
            self._log_progress("COMPLETION", "Finalizing results", 0.95)
            self._save_state()
            
            results = {
                "request_id": self.current_request_id,
                "papers": papers,
                "summaries": summaries,
                "syntheses": syntheses,
                "audio_files": audio_files,
                "citations": citations,
                "classified_papers": classified_papers,
                "processing_time": datetime.now().isoformat()
            }
            
            self._log_progress("COMPLETED", f"Successfully processed {len(papers)} papers", 1.0)
            
            return results
            
        except Exception as e:
            print(f"Error in process_request: {str(e)}")
            return self._create_error_response(f"System error: {str(e)}")
    
    def _collect_papers(self, query: Optional[str], filters: Optional[Dict], 
                       files: Optional[List], urls: Optional[List[str]], 
                       dois: Optional[List[str]]) -> List[Dict[str, Any]]:
        """
        Collect papers from all available sources.
        
        Args:
            query: Search query for arXiv
            filters: Search filters
            files: Uploaded PDF files
            urls: Paper URLs
            dois: DOI references
            
        Returns:
            List of processed paper objects
        """
        papers = []
        
        # Search arXiv if query provided
        if query:
            print("Searching arXiv for papers...")
            arxiv_papers = self.search_agent.search_arxiv(query, filters)
            papers.extend(arxiv_papers)
            print(f"Found {len(arxiv_papers)} papers from arXiv")
        
        # Process uploaded files
        if files:
            print(f"Processing {len(files)} uploaded files...")
            for file in files:
                try:
                    processed_paper = self.processing_agent.process_pdf(file)
                    if processed_paper:
                        papers.append(processed_paper)
                except Exception as e:
                    print(f"Error processing file {file.name}: {str(e)}")
        
        # Process URLs
        if urls:
            print(f"Processing {len(urls)} URLs...")
            for url in urls:
                if url.strip():
                    try:
                        processed_paper = self.processing_agent.process_url(url.strip())
                        if processed_paper:
                            papers.append(processed_paper)
                    except Exception as e:
                        print(f"Error processing URL {url}: {str(e)}")
        
        # Process DOIs
        if dois:
            print(f"Processing {len(dois)} DOIs...")
            for doi in dois:
                if doi.strip():
                    try:
                        processed_paper = self.processing_agent.process_doi(doi.strip())
                        if processed_paper:
                            papers.append(processed_paper)
                    except Exception as e:
                        print(f"Error processing DOI {doi}: {str(e)}")
        
        return papers
    
    def _classify_papers(self, papers: List[Dict[str, Any]], topics: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify papers by user-defined topics.
        
        Args:
            papers: List of paper objects to classify
            topics: List of topic categories
            
        Returns:
            Dictionary mapping topics to lists of papers
        """
        print("Classifying papers by topic...")
        classified_papers = {}
        
        for paper in papers:
            try:
                classification = self.topic_agent.classify_paper(paper, topics)
                topic = classification["assigned_topic"]
                paper["topic"] = topic
                
                if topic not in classified_papers:
                    classified_papers[topic] = []
                classified_papers[topic].append(paper)
                
                self.current_request["classifications"][paper["id"]] = topic
            except Exception as e:
                print(f"Error classifying paper {paper['title']}: {str(e)}")
                # Default to first topic if classification fails
                topic = topics[0]
                paper["topic"] = topic
                if topic not in classified_papers:
                    classified_papers[topic] = []
                classified_papers[topic].append(paper)
        
        return classified_papers
    
    def _generate_summaries(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive summaries for all papers.
        
        Args:
            papers: List of paper objects to summarize
            
        Returns:
            List of summary objects with comprehensive content
        """
        summaries = []
        
        for i, paper in enumerate(papers, 1):
            try:
                progress = 0.4 + (0.2 * i / len(papers))  # 40% to 60%
                self._log_progress("SUMMARIZATION", f"Processing paper {i}/{len(papers)}", 
                                 progress, paper['title'])
                
                comprehensive_summary = self.summary_agent.generate_comprehensive_summary(paper)
                
                summary_obj = {
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper.get("authors", ["Unknown Author"]),
                    "summary": comprehensive_summary,
                    "citation": paper.get("citation", ""),
                    "source": paper.get("source", ""),
                    "url": paper.get("url", "")
                }
                summaries.append(summary_obj)
                self.current_request["summaries"][paper["id"]] = summary_obj
                
                # Print summary statistics - get data from actual summary content
                if isinstance(comprehensive_summary, dict):
                    # Get metadata if available
                    metadata = comprehensive_summary.get("metadata", {})
                    text_source = metadata.get("text_source", "direct_processing")
                    compression = metadata.get("compression_ratio", 1.0)
                    original_length = metadata.get("original_length", 0)
                    
                    # Get actual summary text length
                    summary_text = comprehensive_summary.get("main_summary", "")
                    if not summary_text:
                        summary_text = str(comprehensive_summary)
                    summary_length = len(summary_text)
                else:
                    # Handle string summaries
                    text_source = "direct_processing"
                    summary_length = len(str(comprehensive_summary))
                    compression = 1.0  # No compression data available
                    original_length = 0
                
                print(f"[{i}/{len(papers)}] Summary generated from {text_source}")
                if original_length > 0:
                    print(f"   Length: {summary_length} chars (from {original_length} chars), Compression: {compression:.1f}x")
                else:
                    print(f"   Length: {summary_length} chars, Compression: N/A")
                
            except Exception as e:
                print(f"[{i}/{len(papers)}] Error generating summary for {paper['title']}: {str(e)}")
                # Create fallback summary
                fallback_summary = self._create_fallback_summary(paper, str(e))
                summary_obj = {
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper.get("authors", ["Unknown Author"]),
                    "summary": fallback_summary,
                    "citation": paper.get("citation", ""),
                    "source": paper.get("source", ""),
                    "url": paper.get("url", "")
                }
                summaries.append(summary_obj)
                self.current_request["summaries"][paper["id"]] = summary_obj
        
        print(f"Completed {len(summaries)} paper summaries with enhanced processing")
        return summaries
    
    def _generate_syntheses(self, classified_papers: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Generate topic syntheses for categories with multiple papers.
        
        Args:
            classified_papers: Dictionary mapping topics to paper lists
            
        Returns:
            List of synthesis objects
        """
        print("Generating topic syntheses...")
        syntheses = []
        
        for topic, topic_papers in classified_papers.items():
            if len(topic_papers) > 1:  # Only synthesize if multiple papers
                try:
                    synthesis_text = self.synthesis_agent.synthesize_papers(topic_papers, topic)
                    synthesis_obj = {
                        "topic": topic,
                        "synthesis": synthesis_text,
                        "paper_count": len(topic_papers),
                        "paper_ids": [p["id"] for p in topic_papers]
                    }
                    syntheses.append(synthesis_obj)
                    self.current_request["syntheses"][topic] = synthesis_obj
                except Exception as e:
                    print(f"Error generating synthesis for topic {topic}: {str(e)}")
        
        return syntheses
    
    def _generate_audio_content(self, summaries: List[Dict[str, Any]], 
                               syntheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate podcast-style audio content for summaries and syntheses.
        
        Args:
            summaries: List of summary objects
            syntheses: List of synthesis objects
            
        Returns:
            List of audio file objects with metadata
        """
        audio_files = []
        total_items = len(summaries) + len(syntheses)
        current_item = 0
        
        # Generate podcast audio for summaries
        for summary in summaries:
            try:
                current_item += 1
                progress = 0.75 + (0.1 * current_item / total_items)  # 75% to 85%
                self._log_progress("AUDIO", f"Generating podcast {current_item}/{total_items}", 
                                 progress, summary['title'])
                
                paper_for_podcast = self._prepare_paper_for_podcast(summary)
                podcast_script = self.podcast_agent.generate_podcast_script(paper_for_podcast)
                audio_path = self.podcast_agent.generate_audio_from_script(
                    podcast_script, 
                    f"podcast_{summary['paper_id']}"
                )
                
                if audio_path:
                    audio_obj = {
                        "content_id": summary["paper_id"],
                        "content_type": "podcast",
                        "file_path": audio_path,
                        "script": podcast_script
                    }
                    audio_files.append(audio_obj)
                    self.current_request["audio_files"][summary["paper_id"]] = audio_path
                else:
                    print(f"Audio path is None/empty - podcast generation failed")
                    print(f"Failed to generate podcast audio for {summary['title']}")
                    
            except Exception as e:
                print(f"Error generating podcast for {summary['title']}: {str(e)}")
        
        # Generate podcast audio for syntheses  
        for synthesis in syntheses:
            try:
                print(f"Generating synthesis podcast for topic: {synthesis['topic']}")
                
                synthesis_for_podcast = self._prepare_synthesis_for_podcast(synthesis)
                podcast_script = self.podcast_agent.generate_podcast_script(synthesis_for_podcast)
                audio_path = self.podcast_agent.generate_audio_from_script(
                    podcast_script,
                    f"synthesis_{synthesis['topic'].replace(' ', '_')}"
                )
                
                if audio_path:
                    audio_obj = {
                        "content_id": f"synthesis_{synthesis['topic']}",
                        "content_type": "synthesis_podcast",
                        "file_path": audio_path,
                        "script": podcast_script
                    }
                    audio_files.append(audio_obj)
                    self.current_request["audio_files"][f"synthesis_{synthesis['topic']}"] = audio_path
                    print(f"Synthesis podcast generated: {audio_path}")
                else:
                    print(f"Failed to generate synthesis podcast for {synthesis['topic']}")
                    
            except Exception as e:
                print(f"Error generating synthesis podcast for {synthesis['topic']}: {str(e)}")
        
        return audio_files
    
    def _generate_citations(self, papers: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate formatted citations for all papers.
        
        Args:
            papers: List of paper objects
            
        Returns:
            Dictionary mapping paper IDs to formatted citations
        """
        print("Generating citations...")
        citations = {}
        
        for paper in papers:
            try:
                citation = self.citation_manager.format_citation(paper)
                citations[paper["id"]] = citation
                self.current_request["citations"][paper["id"]] = citation
            except Exception as e:
                print(f"Error generating citation for {paper['title']}: {str(e)}")
        
        return citations
    
    def _create_fallback_summary(self, paper: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Create a fallback summary when summary generation fails.
        
        Args:
            paper: Paper object that failed to summarize
            error_message: Error message to include
            
        Returns:
            Fallback summary object
        """
        return {
            "main_summary": f"Error generating summary for '{paper['title']}': {error_message}",
            "key_points": ["Summary generation failed", "Please try processing again"],
            "executive_summary": f"Unable to process '{paper['title']}' due to processing error",
            "technical_summary": f"Error processing '{paper['title']}': {error_message}",
            "metadata": {"text_source": "error", "compression_ratio": 0, "summary_length": 0}
        }
    
    def _prepare_paper_for_podcast(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a paper summary for podcast generation.
        
        Args:
            summary: Summary object to convert
            
        Returns:
            Paper object formatted for podcast generation
        """
        print(f"Preparing summary for podcast...")
        print(f"Summary keys: {list(summary.keys())}")
        
        if "summary" in summary:
            summary_content = summary["summary"]
            print(f"Summary content type: {type(summary_content)}")
            if isinstance(summary_content, dict):
                main_summary = summary_content.get("main_summary", "")
                print(f"Main summary length: {len(main_summary)}")
            else:
                main_summary = str(summary_content)
                print(f"Summary as string length: {len(main_summary)}")
        else:
            main_summary = ""
            print("No 'summary' key found in summary object")
        
        return {
            "title": summary["title"],
            "authors": summary["authors"],
            "summary": summary["summary"],
            "content": main_summary
        }
    
    def _prepare_synthesis_for_podcast(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a synthesis for podcast generation.
        
        Args:
            synthesis: Synthesis object to convert
            
        Returns:
            Paper object formatted for podcast generation
        """
        return {
            "title": f"Research Synthesis: {synthesis['topic']}",
            "authors": ["Research Team"],
            "summary": {"main_summary": synthesis["synthesis"]},
            "content": synthesis["synthesis"]
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message: Error message to include
            
        Returns:
            Error response dictionary
        """
        return {
            "error": error_message,
            "papers": [],
            "summaries": [],
            "syntheses": [],
            "audio_files": [],
            "citations": {}
        }
    
    def _reset_state(self):
        """
        Reset the current request state for a new processing session.
        
        Clears all stored data from previous requests to ensure clean state
        for new research operations.
        """
        self.current_request = {
            "papers": {},
            "classifications": {},
            "summaries": {},
            "syntheses": {},
            "audio_files": {},
            "citations": {}
        }
    
    def _save_state(self):
        """
        Persist current request state to file for potential recovery or analysis.
        
        Saves the complete state including papers, summaries, syntheses, and metadata
        to a JSON file named with the current request ID.
        """
        if self.current_request_id:
            try:
                state_file = f"{Config.STATE_DIR}/{self.current_request_id}.json"
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(self.current_request, f, indent=2, ensure_ascii=False)
                print(f"State saved to {state_file}")
            except Exception as e:
                print(f"Error saving state: {str(e)}")
    
    def load_state(self, request_id: str) -> Optional[Dict]:
        """
        Load a previously saved request state from file.
        
        Args:
            request_id: Unique identifier of the request to load
            
        Returns:
            Dictionary containing the saved state, or None if not found
        """
        try:
            state_file = f"{Config.STATE_DIR}/{request_id}.json"
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"State file not found for request {request_id}")
            return None
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            return None
