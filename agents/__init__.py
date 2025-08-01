"""Agents package initialization."""

# Import agents directly when needed to avoid circular imports
# from agents.search_agent import SearchAgent
# from agents.processing_agent import ProcessingAgent
# from agents.topic_agent import TopicClassificationAgent
# from agents.summary_agent import SummaryAgent
# from agents.synthesis_agent import SynthesisAgent
# from agents.citation_agent import CitationManager

__all__ = [
    "SearchAgent",
    "ProcessingAgent", 
    "TopicClassificationAgent",
    "SummaryAgent",
    "SynthesisAgent",
    "CitationManager"
]
