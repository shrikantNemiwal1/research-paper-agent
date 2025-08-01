"""
Large Language Model Service

This module provides a unified interface for interacting with multiple LLM providers
including Google Gemini and OpenAI GPT models. It handles API authentication,
request routing, error handling, and fallback mechanisms.

Classes:
    LLMService: Main service class for LLM operations

Features:
    - Multi-provider support (Gemini preferred, OpenAI fallback)
    - Error handling and retry logic
    - Mock responses for development without API keys
    - Text generation, classification, and summarization
    - Configurable generation parameters

Dependencies:
    - google.generativeai: Gemini API integration
    - openai: OpenAI API integration
    - system.config: Configuration management
"""

import google.generativeai as genai
from openai import OpenAI
from typing import Dict, Any, Optional
from system.config import Config


class LLMService:
    """
    Service for Large Language Model operations across multiple providers.
    
    Provides text generation, classification, and summarization using Google Gemini 
    or OpenAI GPT models. Implements provider selection, error handling, and fallback mechanisms.
    
    Attributes:
        config: Configuration object containing API settings
        use_gemini: Boolean indicating if Gemini API is available
        use_openai: Boolean indicating if OpenAI API is available
        gemini_model: Initialized Gemini model instance
        openai_client: Initialized OpenAI client instance
    
    Methods:
        generate_text: Generate text using available LLM provider
        classify_text: Classify text into predefined categories
        summarize_text: Generate summaries of research content
        synthesize_papers: Create cross-paper synthesis analysis
        
    Private Methods:
        _generate_with_gemini: Gemini-specific text generation
        _generate_with_openai: OpenAI-specific text generation
        _generate_mock_response: Fallback mock responses
        _validate_api_response: Response validation and sanitization
    """
    
    def __init__(self):
        """
        Initialize the LLM service with available API configurations.
        
        Configures Gemini API first, falls back to OpenAI if unavailable,
        and provides mock responses if no APIs are configured.
        
        Raises:
            Warning: If no API keys are found, mock responses will be used
        """
        self.config = Config()
        self.use_gemini = bool(self.config.GEMINI_API_KEY)
        self.use_openai = bool(self.config.OPENAI_API_KEY)
        
        self._initialize_providers()
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """
        Initialize available LLM providers based on configured API keys.
        
        Sets up Gemini API first (preferred), then OpenAI as fallback.
        Sets up mock responses if no valid API keys are available.
        """
        if self.use_gemini:
            try:
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                self.use_gemini = False
                self.gemini_model = None
        
        if not self.use_gemini and self.use_openai:
            try:
                self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            except Exception as e:
                self.use_openai = False
                self.openai_client = None
        
        if not self.use_gemini and not self.use_openai:
            self.gemini_model = None
            self.openai_client = None
    
    def generate_text(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """
        Generate text using the available LLM provider with fallback.
        
        Uses Gemini API first, falls back to OpenAI if unavailable,
        and provides mock responses if no APIs are configured. Handles errors
        gracefully and provides meaningful fallback content.
        
        Args:
            prompt: Input text prompt for generation
            max_tokens: Maximum number of tokens to generate (default: 2000)
            temperature: Randomness parameter (0.0-1.0, default: 0.7)
            
        Returns:
            Generated text string, or mock response if APIs unavailable
            
        Raises:
            No exceptions - all errors are handled gracefully with fallbacks
        """
        if self.use_gemini:
            return self._generate_with_gemini(prompt, max_tokens, temperature)
        elif self.use_openai:
            return self._generate_with_openai(prompt, max_tokens, temperature)
        else:
            return self._generate_mock_response(prompt, "no_api_key")
    
    def _generate_with_gemini(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Generate text using Google Gemini API.
        
        Handles Gemini-specific API calls with proper error handling for quota
        limits, billing issues, and general API errors.
        
        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Generation randomness parameter
            
        Returns:
            Generated text or error message
        """
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Add timeout handling to prevent hanging (Windows-compatible)
            import threading
            import time
            
            response_container = [None]
            exception_container = [None]
            
            def api_call():
                try:
                    response_container[0] = self.gemini_model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                except Exception as e:
                    exception_container[0] = e
            
            # Start API call in separate thread with 30 second timeout
            thread = threading.Thread(target=api_call)
            thread.daemon = True
            thread.start()
            thread.join(timeout=30)
            
            if thread.is_alive():
                return "API call timed out - please try again"
            
            if exception_container[0]:
                raise exception_container[0]
                
            response = response_container[0]
            if response is None:
                return "API call failed - please try again"
            
            if response.text:
                return response.text.strip()
            else:
                return "No response generated"
                
        except Exception as e:
            error_msg = str(e)
            
            if self._is_quota_error(error_msg):
                return self._generate_mock_response(prompt, "quota_exceeded")
            else:
                return f"Error generating text with Gemini: {error_msg}"
    
    def _generate_with_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Generate text using OpenAI GPT API as fallback.
        
        Handles OpenAI-specific API calls with proper error handling for quota
        limits, billing issues, and general API errors.
        
        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Generation randomness parameter
            
        Returns:
            Generated text or error message
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            else:
                return "No response generated"
                
        except Exception as e:
            error_msg = str(e)
            
            if self._is_quota_error(error_msg):
                return self._generate_mock_response(prompt, "quota_exceeded")
            else:
                return f"Error generating text with OpenAI: {error_msg}"
    
    def _is_quota_error(self, error_message: str) -> bool:
        """
        Check if an error message indicates quota or billing issues.
        
        Args:
            error_message: Error message to analyze
            
        Returns:
            True if error is related to quota/billing limits
        """
        error_lower = error_message.lower()
        quota_indicators = ["quota", "billing", "limit", "exceeded", "rate limit"]
        return any(indicator in error_lower for indicator in quota_indicators)
    
    def _generate_mock_response(self, prompt: str, error_type: str = "general") -> str:
        """
        Generate contextually appropriate mock responses for development and testing.
        
        Provides meaningful placeholder content based on the type of request
        when LLM APIs are unavailable or have exceeded quotas.
        
        Args:
            prompt: Original prompt to analyze for response type
            error_type: Type of error triggering mock response
            
        Returns:
            Contextually appropriate mock response text
        """
        prompt_lower = prompt.lower()
        
        if "summary" in prompt_lower or "summarize" in prompt_lower:
            return self._generate_mock_summary(error_type)
        elif "classify" in prompt_lower:
            return "Machine Learning"  # Default classification
        elif "synthesis" in prompt_lower or "synthesize" in prompt_lower:
            return self._generate_mock_synthesis(error_type)
        else:
            return self._generate_generic_mock_response(error_type)
    
    def _generate_mock_summary(self, error_type: str) -> str:
        """Generate mock summary response."""
        base_summary = """This is a mock summary generated because the LLM API is unavailable. 
        In a real implementation with Gemini or OpenAI, this would contain an AI-generated summary including:
        1. Main research objectives and questions
        2. Methodology and approach used
        3. Key findings and results  
        4. Conclusions and implications"""
        
        if error_type == "quota_exceeded":
            return base_summary + "\n\nAPI quota exceeded. Please check your billing and usage limits."
        else:
            return base_summary + "\n\nPlease set GEMINI_API_KEY or OPENAI_API_KEY to enable real AI summaries."
    
    def _generate_mock_synthesis(self, error_type: str) -> str:
        """Generate mock synthesis response."""
        base_synthesis = """This is a mock synthesis generated because the LLM API is unavailable.
        In a real implementation with Gemini or OpenAI, this would analyze multiple papers and provide:
        1. Common themes across the research papers
        2. Contradictory findings and different approaches
        3. Emerging trends in the research area
        4. Identified gaps and future research opportunities"""
        
        if error_type == "quota_exceeded":
            return base_synthesis + "\n\nAPI quota exceeded. Please check your billing and usage limits."
        else:
            return base_synthesis + "\n\nPlease set GEMINI_API_KEY or OPENAI_API_KEY to enable real AI synthesis."
    
    def _generate_generic_mock_response(self, error_type: str) -> str:
        """Generate generic mock response."""
        if error_type == "no_api_key":
            return "Mock response: No API keys configured. Please set GEMINI_API_KEY or OPENAI_API_KEY."
        else:
            return f"Mock response generated due to API {error_type}. Please check your API quota and billing."
    
    def classify_text(self, text: str, categories: list, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Classify text content into one of the provided categories.
        
        Uses LLM to analyze text content and assign it to the most appropriate
        category from the provided list. Includes confidence scoring and
        fallback handling for classification failures.
        
        Args:
            text: Text content to classify
            categories: List of possible category names
            temperature: Generation randomness (lower for more deterministic results)
            
        Returns:
            Dictionary containing:
                - assigned_topic: Selected category name
                - confidence: Confidence score (0.0-1.0)
        """
        if not categories:
            return {"assigned_topic": "General", "confidence": 0.0}
        
        prompt = self._build_classification_prompt(text, categories)
        
        try:
            generated_category = self.generate_text(
                prompt, 
                max_tokens=Config.MAX_CLASSIFICATION_TOKENS, 
                temperature=temperature
            )
            
            return self._parse_classification_result(generated_category, categories)
            
        except Exception as e:
            return {
                "assigned_topic": categories[0],
                "confidence": 0.0
            }
    
    def summarize_text(self, text: str, title: str = "", authors: str = "") -> str:
        """
        Generate a comprehensive summary of research paper content.
        
        Creates a structured summary including research objectives, methodology,
        key findings, and implications. Handles text length limits and provides
        contextual information when available.
        
        Args:
            text: Main content to summarize
            title: Optional paper title for context
            authors: Optional author information for context
            
        Returns:
            Comprehensive summary text
        """
        # Validate and prepare input text
        processed_text = self._prepare_text_for_processing(text)
        
        prompt = self._build_summarization_prompt(processed_text, title, authors)
        
        return self.generate_text(
            prompt, 
            max_tokens=Config.MAX_SUMMARY_TOKENS, 
            temperature=Config.TEMPERATURE_SUMMARY
        )
    
    def synthesize_papers(self, papers: list, topic: str) -> str:
        """
        Generate a cross-paper synthesis analysis for a specific topic.
        
        Analyzes multiple research papers to identify common themes, contradictory
        findings, emerging trends, and research gaps within a topic area.
        
        Args:
            papers: List of paper objects to synthesize
            topic: Topic area for focused synthesis
            
        Returns:
            Comprehensive synthesis text analyzing patterns across papers
        """
        if not papers:
            return f"No papers available for synthesis on topic: {topic}"
        
        papers_text = self._prepare_papers_for_synthesis(papers)
        prompt = self._build_synthesis_prompt(papers_text, topic)
        
        return self.generate_text(
            prompt, 
            max_tokens=Config.MAX_SYNTHESIS_TOKENS, 
            temperature=Config.TEMPERATURE_SYNTHESIS
        )
    
    def _build_classification_prompt(self, text: str, categories: list) -> str:
        """
        Build a well-structured prompt for text classification.
        
        Args:
            text: Text to classify
            categories: Available categories
            
        Returns:
            Formatted classification prompt
        """
        return f"""Classify the following text into one of these categories: {', '.join(categories)}.

Text: {text}

Return only the name of the most appropriate category from the list."""
    
    def _parse_classification_result(self, generated_category: str, categories: list) -> Dict[str, Any]:
        """
        Parse and validate classification results.
        
        Args:
            generated_category: Raw classification result
            categories: Valid category list
            
        Returns:
            Parsed classification with confidence score
        """
        # Check for exact or partial matches
        for category in categories:
            if category.lower() in generated_category.lower():
                return {
                    "assigned_topic": category,
                    "confidence": 0.8
                }
        
        # Return default if no match found
        return {
            "assigned_topic": categories[0],
            "confidence": 0.5
        }
    
    def _prepare_text_for_processing(self, text: str) -> str:
        """
        Prepare and validate text for LLM processing.
        
        Args:
            text: Raw text content
            
        Returns:
            Processed text within length limits
        """
        if not text:
            return "No content available"
        
        if len(text) > Config.MAX_TEXT_LENGTH:
            return text[:Config.MAX_TEXT_LENGTH] + "..."
        
        return text
    
    def _build_summarization_prompt(self, text: str, title: str, authors: str) -> str:
        """
        Build a comprehensive summarization prompt.
        
        Args:
            text: Content to summarize
            title: Paper title
            authors: Author information
            
        Returns:
            Formatted summarization prompt
        """
        return f"""Summarize the following research paper. Include key findings, methodology, and implications.

Title: {title}
Authors: {authors}
Content: {text}

Provide a comprehensive summary in 3-5 paragraphs. Include:
1. The main research question and objectives
2. The methodology used
3. Key findings and results
4. Implications and conclusions

Summary:"""
    
    def _prepare_papers_for_synthesis(self, papers: list) -> str:
        """
        Prepare multiple papers for synthesis analysis.
        
        Args:
            papers: List of paper objects
            
        Returns:
            Formatted text combining paper information
        """
        paper_summaries = []
        
        for paper in papers:
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "")
            content = paper.get("content", {}).get("full_text", "")
            
            # Use abstract if available, otherwise truncate content
            summary_text = abstract if abstract else content[:500]
            paper_summaries.append(f"Title: {title}\nSummary: {summary_text}")
        
        papers_text = "\n\n".join(paper_summaries)
        
        # Ensure within length limits
        if len(papers_text) > Config.MAX_TEXT_LENGTH:
            papers_text = papers_text[:Config.MAX_TEXT_LENGTH] + "..."
        
        return papers_text
    
    def _build_synthesis_prompt(self, papers_text: str, topic: str) -> str:
        """
        Build a structured prompt for cross-paper synthesis.
        
        Args:
            papers_text: Combined paper content
            topic: Focus topic for synthesis
            
        Returns:
            Formatted synthesis prompt
        """
        return f"""Generate a synthesis of the following research papers on the topic "{topic}".
Identify common themes, approaches, contradictions, and potential research gaps.

Papers:
{papers_text}

Your synthesis should:
1. Identify common themes across the papers
2. Highlight any contradictory findings
3. Note emerging trends in this research area
4. Identify potential gaps or opportunities for future research

Synthesis:"""
