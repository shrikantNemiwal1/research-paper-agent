"""
Topic Classification Agent

This module provides topic classification capabilities for research papers,
using language models to categorize papers into user-defined topics with
confidence scoring and batch processing support.

Classes:
    TopicClassificationAgent: Main agent class for topic classification

Features:
    - Topic classification using LLM services
    - Confidence scoring for classification results
    - Batch processing for multiple papers
    - Topic distribution analysis and reporting
    - Fallback classification mechanisms
    - Text preprocessing for classification

Dependencies:
    - services.llm_service: Language model service integration
    - typing: Type hints and annotations
"""

from typing import Dict, Any, List
from services.llm_service import LLMService


class TopicClassificationAgent:
    """
    Topic Classification Agent
    
    Agent for categorizing research papers into user-defined topics using
    language model analysis and content understanding.
    
    This agent provides:
    - Topic classification with confidence scoring
    - Text analysis and content extraction
    - Batch processing for multiple papers
    - Topic distribution analysis and reporting
    - Fallback mechanisms for classification failures
    - Text preprocessing for classification
    
    Attributes:
        llm_service (LLMService): Language model service for classification
        
    Methods:
        classify_paper(paper, topics): Classify single paper into topics
        classify_multiple_papers(papers, topics): Batch paper classification
        get_topic_distribution(classifications): Analyze topic distribution
        _prepare_classification_text(paper): Prepare text for classification
        _validate_classification_inputs(paper, topics): Validate inputs
        _create_fallback_classification(topics): Create fallback result
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the topic classification agent.
        
        Args:
            llm_service (LLMService): Language model service for text classification.
        """
        self.llm_service = llm_service
    
    def classify_paper(self, paper: Dict[str, Any], topics: List[str]) -> Dict[str, Any]:
        """
        Classify research paper into most appropriate topic category.
        
        Args:
            paper (Dict[str, Any]): Paper object containing title, abstract, and content.
            topics (List[str]): List of user-defined topic categories.
            
        Returns:
            Dict[str, Any]: Classification result with assigned topic and confidence score.
        """
        print(f"TopicClassificationAgent: Classifying '{paper.get('title', 'Unknown')}'")
        
        try:
            # Validate inputs
            if not self._validate_classification_inputs(paper, topics):
                return self._create_fallback_classification(topics)
            
            # Prepare text for classification
            classification_text = self._prepare_classification_text(paper)
            
            # Perform classification using LLM service
            classification_result = self.llm_service.classify_text(classification_text, topics)
            
            assigned_topic = classification_result.get('assigned_topic', topics[0])
            print(f"TopicClassificationAgent: Classified as '{assigned_topic}'")
            return classification_result
            
        except Exception as e:
            print(f"TopicClassificationAgent error: {str(e)}")
            return self._create_fallback_classification(topics)
    
    def classify_multiple_papers(self, papers: List[Dict[str, Any]], topics: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Classify multiple research papers into topic categories in batch.
        
        Args:
            papers (List[Dict[str, Any]]): List of paper objects for classification.
            topics (List[str]): List of user-defined topic categories.
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping paper IDs to classification results.
        """
        print(f"TopicClassificationAgent: Classifying {len(papers)} papers into {len(topics)} topics")
        
        classifications = {}
        successful_count = 0
        
        for paper in papers:
            try:
                paper_id = paper.get("id", "unknown")
                classification = self.classify_paper(paper, topics)
                classifications[paper_id] = classification
                successful_count += 1
            except Exception as e:
                print(f"Error classifying paper {paper.get('title', 'Unknown')}: {str(e)}")
                classifications[paper.get("id", "unknown")] = self._create_fallback_classification(topics)
        
        print(f"TopicClassificationAgent: Successfully classified {successful_count}/{len(papers)} papers")
        return classifications
    
    def get_topic_distribution(self, classifications: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze topic distribution across classified papers.
        
        Args:
            classifications (Dict[str, Dict[str, Any]]): Classification results from paper classification.
            
        Returns:
            Dict[str, int]: Dictionary mapping topic names to paper counts.
        """
        distribution = {}
        
        for classification in classifications.values():
            topic = classification.get("assigned_topic", "Unknown")
            distribution[topic] = distribution.get(topic, 0) + 1
        
        return distribution
    
    def _validate_classification_inputs(self, paper: Dict[str, Any], topics: List[str]) -> bool:
        """
        Validate inputs for classification process.
        
        Args:
            paper (Dict[str, Any]): Paper object to validate
            topics (List[str]): Topics list to validate
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        if not paper or not isinstance(paper, dict):
            print("TopicClassificationAgent: Invalid paper object")
            return False
            
        if not topics or not isinstance(topics, list) or not topics:
            print("TopicClassificationAgent: Invalid or empty topics list")
            return False
            
        return True
    
    def _prepare_classification_text(self, paper: Dict[str, Any]) -> str:
        """
        Prepare text for classification analysis.
        
        Args:
            paper (Dict[str, Any]): Paper object with content.
            
        Returns:
            str: Text formatted for classification.
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Prefer abstract for classification accuracy
        if abstract and abstract.strip():
            return f"Title: {title}\nAbstract: {abstract}"
        else:
            # Use truncated content as fallback
            content = paper.get("content", {}).get("full_text", "")
            truncated_content = content[:1000] if content else ""
            return f"Title: {title}\nContent: {truncated_content}"
    
    def _create_fallback_classification(self, topics: List[str]) -> Dict[str, Any]:
        """
        Create fallback classification result.
        
        Args:
            topics (List[str]): Available topics
            
        Returns:
            Dict[str, Any]: Fallback classification result
        """
        return {
            "assigned_topic": topics[0] if topics else "General",
            "confidence": 0.0
        }
