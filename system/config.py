"""Configuration settings for the Research Paper Multi-Agent System."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """System configuration class."""
    
    # API Keys - Set these as environment variables
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    # Keeping OpenAI for backward compatibility (optional)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # System settings
    MAX_PAPERS_PER_SEARCH = 10
    MAX_TEXT_LENGTH = 4000  # Max text length for LLM processing
    MAX_SUMMARY_TOKENS = 500
    MAX_SYNTHESIS_TOKENS = 800
    MAX_CLASSIFICATION_TOKENS = 50
    
    # Temperature settings for different LLM tasks
    TEMPERATURE_CLASSIFICATION = 0.3
    TEMPERATURE_SUMMARY = 0.5
    TEMPERATURE_SYNTHESIS = 0.5
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one more level
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
    AUDIO_DIR = os.path.join(DATA_DIR, "audio")
    STATE_DIR = os.path.join(DATA_DIR, "state")
    
    # Ensure directories exist
    for directory in [DATA_DIR, UPLOADS_DIR, AUDIO_DIR, STATE_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # arXiv search settings
    ARXIV_MAX_RESULTS = 10
    ARXIV_SORT_BY = "relevance"  # or "lastUpdatedDate"
    
    # PDF processing settings
    PDF_MAX_PAGES = 20  # Limit for processing large PDFs
    
    # Audio settings
    AUDIO_LANGUAGE = "en"
    AUDIO_SLOW = False
    
    # Default topics if none provided
    DEFAULT_TOPICS = [
        "Machine Learning",
        "Natural Language Processing", 
        "Computer Vision",
        "Data Science",
        "Artificial Intelligence"
    ]
    
    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get Gemini configuration."""
        return {
            "api_key": cls.GEMINI_API_KEY,
            "model": "gemini-1.5-flash",
            "max_tokens": 500,
            "temperature": 0.5
        }
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI configuration (for backward compatibility)."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.5
        }
    
    @classmethod
    def get_arxiv_config(cls) -> Dict[str, Any]:
        """Get arXiv configuration."""
        return {
            "max_results": cls.ARXIV_MAX_RESULTS,
            "sort_by": cls.ARXIV_SORT_BY
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.GEMINI_API_KEY and not cls.OPENAI_API_KEY:
            print("Warning: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set. Please set at least one environment variable.")
            return False
        
        if cls.GEMINI_API_KEY:
            print("Using Gemini API")
            return True
        elif cls.OPENAI_API_KEY:
            print("Using OpenAI API")
            return True
        
        return True
