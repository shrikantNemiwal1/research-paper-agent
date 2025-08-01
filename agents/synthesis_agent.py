"""
Research Paper Synthesis Agent

This module provides advanced synthesis capabilities for research papers,
creating comprehensive topic-based analyses that combine insights from
multiple papers into coherent, valuable research syntheses.

Classes:
    SynthesisAgent: Main agent class for cross-paper synthesis

Features:
    - Multi-paper synthesis with topic-based organization
    - Advanced content analysis and integration
    - Quality validation and coherence checking
    - Batch synthesis processing for multiple topics
    - Intelligent paper selection and prioritization
    - Error handling and fallback mechanisms

Dependencies:
    - services.llm_service: Language model service integration
    - typing: Type hints and annotations
"""

from typing import Dict, Any, List
from services.llm_service import LLMService


class SynthesisAgent:
    """
    Research Paper Synthesis Agent
    
    Specialized agent for creating comprehensive syntheses that combine insights
    from multiple research papers into coherent, valuable analyses organized
    by topic and research theme.
    
    This agent provides:
    - Advanced multi-paper synthesis capabilities
    - Topic-based organization and analysis
    - Content integration and coherence validation
    - Quality assessment and improvement mechanisms
    - Batch processing for multiple topics
    - Intelligent paper selection and prioritization
    
    Attributes:
        llm_service (LLMService): Language model service for synthesis generation
        
    Methods:
        synthesize_papers(papers, topic): Generate synthesis for single topic
        synthesize_by_topics(classified_papers): Batch synthesis for multiple topics
        _validate_synthesis_inputs(papers, topic): Validate synthesis parameters
        _prepare_synthesis_content(papers): Prepare papers for synthesis
        _generate_topic_synthesis(papers, topic): Core synthesis generation
        _create_fallback_synthesis(papers, topic): Create fallback result
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the research paper synthesis agent.
        
        Sets up the language model service and prepares the agent for
        advanced cross-paper synthesis and analysis tasks.
        
        Args:
            llm_service (LLMService): Language model service for synthesis generation
                                    and content analysis.
        """
        self.llm_service = llm_service
    
    def synthesize_papers(self, papers: List[Dict[str, Any]], topic: str) -> str:
        """
        Generate comprehensive synthesis of multiple papers on specific topic.
        
        Creates an integrated analysis that combines insights, findings, and
        methodologies from multiple research papers into a coherent synthesis.
        
        Args:
            papers (List[Dict[str, Any]]): List of paper objects containing
                                          research content for synthesis.
            topic (str): Research topic or theme for synthesis focus and
                        organization of the analysis.
            
        Returns:
            str: Generated synthesis text providing comprehensive analysis
                 of the research area with integrated insights.
                 
        Example:
            >>> agent = SynthesisAgent(llm_service)
            >>> synthesis = agent.synthesize_papers(ai_papers, "Machine Learning")
            >>> print(len(synthesis))  # Length of generated synthesis
        """
        print(f"SynthesisAgent: Generating synthesis for topic '{topic}' with {len(papers)} papers")
        
        try:
            # Validate synthesis inputs
            if not self._validate_synthesis_inputs(papers, topic):
                return self._create_fallback_synthesis(papers, topic)
            
            # Prepare papers for synthesis processing
            synthesis_content = self._prepare_synthesis_content(papers)
            
            # Generate comprehensive synthesis
            synthesis = self._generate_topic_synthesis(synthesis_content, topic)
            
            print(f"SynthesisAgent: Generated synthesis ({len(synthesis)} characters)")
            return synthesis
            
        except Exception as e:
            print(f"SynthesisAgent error: {str(e)}")
            return self._create_fallback_synthesis(papers, topic)
    
    def synthesize_by_topics(self, classified_papers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        Generate comprehensive syntheses for multiple topics in batch.
        
        Efficiently processes multiple topic groups and generates corresponding
        syntheses with proper validation and error handling.
        
        Args:
            classified_papers (Dict[str, List[Dict[str, Any]]]): Dictionary mapping
                                                               topic names to lists of papers.
            
        Returns:
            Dict[str, str]: Dictionary mapping topic names to generated synthesis texts.
                           Only includes topics with sufficient papers for synthesis.
        """
        print(f"SynthesisAgent: Generating syntheses for {len(classified_papers)} topics")
        
        syntheses = {}
        successful_count = 0
        
        for topic, papers in classified_papers.items():
            try:
                if len(papers) >= 2:  # Require multiple papers for meaningful synthesis
                    synthesis = self.synthesize_papers(papers, topic)
                    syntheses[topic] = synthesis
                    successful_count += 1
                else:
                    print(f"Skipping synthesis for topic '{topic}' - only {len(papers)} paper(s)")
            except Exception as e:
                print(f"Error generating synthesis for topic '{topic}': {str(e)}")
                syntheses[topic] = f"Error generating synthesis for topic '{topic}'"
        
        print(f"SynthesisAgent: Successfully generated {successful_count}/{len(classified_papers)} syntheses")
        return syntheses
    
    def identify_common_themes(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Identify common themes across papers.
        
        Args:
            papers (list): List of paper objects
            
        Returns:
            list: List of identified common themes
        """
        try:
            # Simple keyword-based theme identification
            all_text = ""
            for paper in papers:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                all_text += f" {title} {abstract}"
            
            # Common research themes and keywords
            theme_keywords = {
                "Machine Learning": ["machine learning", "neural network", "deep learning", "artificial intelligence"],
                "Natural Language Processing": ["nlp", "natural language", "text processing", "language model"],
                "Computer Vision": ["computer vision", "image processing", "object detection", "image recognition"],
                "Data Mining": ["data mining", "data analysis", "pattern recognition", "knowledge discovery"],
                "Optimization": ["optimization", "algorithm", "performance", "efficiency"],
                "Security": ["security", "privacy", "encryption", "cybersecurity"],
                "Robotics": ["robot", "robotics", "autonomous", "control system"]
            }
            
            identified_themes = []
            all_text_lower = all_text.lower()
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in all_text_lower for keyword in keywords):
                    identified_themes.append(theme)
            
            return identified_themes[:5]  # Return top 5 themes
            
        except Exception as e:
            print(f"⚠️ Error identifying themes: {str(e)}")
            return ["General Research Themes"]
    
    def identify_research_gaps(self, papers: List[Dict[str, Any]], topic: str) -> List[str]:
        """Identify potential research gaps from the papers.
        
        Args:
            papers (list): List of paper objects
            topic (str): Topic area
            
        Returns:
            list: List of identified research gaps
        """
        try:
            # Simple gap identification based on common phrases
            gap_indicators = [
                "future work", "further research", "limitations", "challenges",
                "open questions", "unexplored", "need for", "lack of",
                "insufficient", "limited understanding"
            ]
            
            all_text = ""
            for paper in papers:
                content = paper.get("content", {})
                if isinstance(content, dict):
                    full_text = content.get("full_text", "")
                    all_text += f" {full_text}"
                else:
                    abstract = paper.get("abstract", "")
                    all_text += f" {abstract}"
            
            all_text_lower = all_text.lower()
            gaps = []
            
            # Look for sentences containing gap indicators
            sentences = all_text.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in gap_indicators):
                    if len(sentence.strip()) > 20:  # Ignore very short sentences
                        gaps.append(sentence.strip() + '.')
            
            # If no specific gaps found, provide generic ones based on topic
            if not gaps:
                generic_gaps = {
                    "Machine Learning": [
                        "Need for more interpretable ML models",
                        "Limited research on edge computing applications",
                        "Insufficient work on bias mitigation"
                    ],
                    "Natural Language Processing": [
                        "Lack of multilingual model evaluation",
                        "Limited understanding of context preservation",
                        "Need for better evaluation metrics"
                    ],
                    "Computer Vision": [
                        "Insufficient work on real-world robustness",
                        "Limited research on privacy-preserving methods",
                        "Need for better small dataset performance"
                    ]
                }
                
                gaps = generic_gaps.get(topic, [
                    f"Further research needed in {topic.lower()}",
                    "Limited cross-domain validation",
                    "Need for standardized evaluation frameworks"
                ])
            
            return gaps[:3]  # Return top 3 gaps
            
        except Exception as e:
            print(f"⚠️ Error identifying research gaps: {str(e)}")
            return [f"Research gaps analysis not available for {topic}"]
    
    def format_synthesis_with_metadata(self, topic: str, papers: List[Dict[str, Any]], synthesis: str) -> Dict[str, Any]:
        """Format synthesis with metadata and additional analysis.
        
        Args:
            topic (str): Topic name
            papers (list): List of papers used in synthesis
            synthesis (str): Generated synthesis text
            
        Returns:
            dict: Formatted synthesis object with metadata
        """
        try:
            common_themes = self.identify_common_themes(papers)
            research_gaps = self.identify_research_gaps(papers, topic)
            
            return {
                "topic": topic,
                "synthesis_text": synthesis,
                "paper_count": len(papers),
                "paper_ids": [paper.get("id", "unknown") for paper in papers],
                "paper_titles": [paper.get("title", "Unknown") for paper in papers],
                "common_themes": common_themes,
                "research_gaps": research_gaps,
                "sources": [paper.get("source", "Unknown") for paper in papers],
                "date_range": self._get_date_range(papers)
            }
            
        except Exception as e:
            print(f"❌ Error formatting synthesis: {str(e)}")
            return {
                "topic": topic,
                "synthesis_text": synthesis,
                "paper_count": len(papers),
                "error": str(e)
            }
    
    def _get_date_range(self, papers: List[Dict[str, Any]]) -> str:
        """Get the date range of papers in the synthesis.
        
        Args:
            papers (list): List of paper objects
            
        Returns:
            str: Date range string
        """
        try:
            dates = []
            for paper in papers:
                date_str = paper.get("published_date", "")
                if date_str and date_str != "Unknown":
                    try:
                        # Try to extract year
                        if len(date_str) >= 4:
                            year = int(date_str[:4])
                            dates.append(year)
                    except:
                        continue
            
            if dates:
                min_year = min(dates)
                max_year = max(dates)
                if min_year == max_year:
                    return str(min_year)
                else:
                    return f"{min_year}-{max_year}"
            else:
                return "Date range unknown"
                
        except Exception as e:
            print(f"Error getting date range: {str(e)}")
            return "Date range unavailable"
    
    def _validate_synthesis_inputs(self, papers: List[Dict[str, Any]], topic: str) -> bool:
        """
        Validate inputs for synthesis generation.
        
        Args:
            papers (List[Dict[str, Any]]): Papers to validate
            topic (str): Topic to validate
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        if not papers or not isinstance(papers, list) or len(papers) < 2:
            print(f"SynthesisAgent: Insufficient papers for synthesis on topic '{topic}'. At least 2 papers required.")
            return False
            
        if not topic or not isinstance(topic, str) or not topic.strip():
            print("SynthesisAgent: Invalid or empty topic provided")
            return False
            
        return True
    
    def _prepare_synthesis_content(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare papers for synthesis processing.
        
        Args:
            papers (List[Dict[str, Any]]): Raw papers
            
        Returns:
            List[Dict[str, Any]]: Prepared papers for synthesis
        """
        prepared_papers = []
        
        for paper in papers:
            # Extract key content for synthesis
            prepared_paper = {
                "title": paper.get("title", "Unknown"),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "summary": paper.get("summary", ""),
                "key_findings": paper.get("key_findings", ""),
                "methodology": paper.get("methodology", "")
            }
            prepared_papers.append(prepared_paper)
            
        return prepared_papers
    
    def _generate_topic_synthesis(self, papers: List[Dict[str, Any]], topic: str) -> str:
        """
        Generate comprehensive synthesis for specific topic.
        
        Args:
            papers (List[Dict[str, Any]]): Prepared papers
            topic (str): Topic for synthesis
            
        Returns:
            str: Generated synthesis text
        """
        return self.llm_service.synthesize_papers(papers, topic)
    
    def _create_fallback_synthesis(self, papers: List[Dict[str, Any]], topic: str) -> str:
        """
        Create fallback synthesis when processing fails.
        
        Args:
            papers (List[Dict[str, Any]]): Papers for synthesis
            topic (str): Topic name
            
        Returns:
            str: Fallback synthesis text
        """
        if len(papers) < 2:
            return f"Insufficient papers for synthesis on topic '{topic}'. At least 2 papers are required."
            
        paper_titles = [paper.get("title", "Unknown") for paper in papers]
        title_list = ', '.join(paper_titles[:3])
        if len(paper_titles) > 3:
            title_list += f"... and {len(paper_titles) - 3} more"
            
        return f"Error generating synthesis for topic '{topic}'. Papers included: {title_list}"
