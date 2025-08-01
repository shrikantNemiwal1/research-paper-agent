"""Main Streamlit application for the Research Paper Multi-Agent System."""

import streamlit as st
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add the current directory to the path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from system.config import Config
    from system.controller import ResearchSystem
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you're running the app from the research_paper_system directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Research Paper Multi-Agent System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.error-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def check_system_configuration():
    """Check if the system is properly configured."""
    try:
        
        # Check if OpenAI API key is set
        if not Config.OPENAI_API_KEY:
            st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.info("You can set it in your environment or create a .env file with: OPENAI_API_KEY=your_key_here")
            
            # Additional debugging
            env_file_path = os.path.join(os.getcwd(), '.env')
            st.write(f"Looking for .env file at: {env_file_path}")
            st.write(f".env file exists: {os.path.exists(env_file_path)}")
            
            if os.path.exists(env_file_path):
                st.write("✅ .env file found - please check if OPENAI_API_KEY is properly set")
            else:
                st.write("❌ .env file not found - please create one with your API key")
            
            return False
        
        # Check if data directories exist
        for directory in [Config.DATA_DIR, Config.UPLOADS_DIR, Config.AUDIO_DIR, Config.STATE_DIR]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    st.error(f"❌ Could not create directory {directory}: {str(e)}")
                    return False
        
        return True
        
    except Exception as e:
        st.error(f"❌ Configuration error: {str(e)}")
        return False

def display_header():
    """Display the main header."""
    st.markdown('<h1 class="main-header">🔬 Research Paper Multi-Agent System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This system helps you search, process, summarize, and synthesize research papers using AI agents.
    You can search arXiv, upload PDFs, provide URLs/DOIs, and get comprehensive summaries with conversational podcast versions.
    </div>
    """, unsafe_allow_html=True)

def display_input_section():
    """Display the input section for research parameters."""
    st.markdown('<h2 class="section-header">📋 Research Input</h2>', unsafe_allow_html=True)
    
    # Research query input
    query = st.text_input(
        "Research Topic/Question:",
        placeholder="e.g., 'transformer models in natural language processing'",
        help="Enter a research topic to search for papers on arXiv"
    )
    
    # Topic definition
    st.markdown("### 🏷️ Define Research Topics")
    st.info("Define 2-3 topics to categorize your papers. These will be used for classification and synthesis.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        topic1 = st.text_input("Topic 1:", placeholder="e.g., Machine Learning")
    with col2:
        topic2 = st.text_input("Topic 2:", placeholder="e.g., Natural Language Processing")
    with col3:
        topic3 = st.text_input("Topic 3:", placeholder="e.g., Computer Vision")
    
    topics = [topic for topic in [topic1, topic2, topic3] if topic.strip()]
    
    # File upload section
    st.markdown("### 📎 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF research papers:",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF files of research papers you want to analyze"
    )
    
    # URL and DOI input
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌐 Paper URLs")
        urls_text = st.text_area(
            "Enter paper URLs (one per line):",
            placeholder="https://arxiv.org/pdf/2301.00001.pdf\nhttps://example.com/paper.pdf",
            help="Enter URLs to research papers (PDFs or web pages)"
        )
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    with col2:
        st.markdown("### 🔗 DOI References")
        dois_text = st.text_area(
            "Enter DOIs (one per line):",
            placeholder="10.1000/182\n10.1038/nature12373",
            help="Enter DOI references to research papers"
        )
        dois = [doi.strip() for doi in dois_text.split('\n') if doi.strip()]
    
    # Search filters
    st.markdown("### ⚙️ Search Filters")
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.slider(
            "Publication Years:",
            min_value=2010,
            max_value=2025,
            value=(2020, 2025),
            help="Filter papers by publication year range"
        )
    with col2:
        max_papers = st.slider(
            "Maximum Papers:",
            min_value=1,
            max_value=20,
            value=2,
            help="Maximum number of papers to retrieve from arXiv search"
        )
    
    filters = {
        "date_range": date_range,
        "max_results": max_papers
    }
    
    return query, topics, uploaded_files, urls, dois, filters

def validate_inputs(query, topics, uploaded_files, urls, dois):
    """Validate user inputs."""
    if not query and not uploaded_files and not urls and not dois:
        st.error("❌ Please provide at least one input: a research query, upload files, enter URLs, or provide DOIs.")
        return False
    
    if not topics:
        st.warning("⚠️ No topics defined. Using default topics for classification.")
    
    return True

def display_processing_section(research_system, query, topics, filters, uploaded_files, urls, dois):
    """Display the processing section and handle research execution."""
    st.markdown('<h2 class="section-header">⚙️ Processing</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Start Research Process", type="primary", use_container_width=True):
        if not validate_inputs(query, topics, uploaded_files, urls, dois):
            return
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_container = st.empty()
        stage_container = st.empty()
        paper_container = st.empty()
        
        # Progress callback function
        def update_progress(progress_data):
            stage = progress_data.get('stage', '')
            message = progress_data.get('message', '')
            progress = progress_data.get('progress', 0)
            paper_title = progress_data.get('paper_title', '')
            
            if progress:
                progress_bar.progress(progress)
            
            stage_container.info(f"� **{stage}**: {message}")
            
            if paper_title:
                paper_container.text(f"📄 Current paper: {paper_title[:60]}...")
            else:
                paper_container.empty()
        
        try:
            # Initialize research system with progress callback
            research_system_with_progress = ResearchSystem(progress_callback=update_progress)
            
            # Process the research request
            results = research_system_with_progress.process_request(
                query=query,
                topics=topics,
                filters=filters,
                files=uploaded_files,
                urls=urls,
                dois=dois
            )
            
            progress_bar.progress(1.0)
            
            if "error" in results:
                status_container.error(f"❌ Error: {results['error']}")
                return
            
            # Store results in session state
            st.session_state.research_results = results
            st.session_state.processing_complete = True
            
            stage_container.success("✅ Research processing completed successfully!")
            paper_container.empty()
            
            # Display summary
            summary_text = f"""
            **Processing Summary:**
            - Papers processed: {len(results.get('papers', []))}
            - Summaries generated: {len(results.get('summaries', []))}
            - Topic syntheses: {len(results.get('syntheses', []))}
            - Podcast files created: {len(results.get('audio_files', []))}
            """
            st.markdown(summary_text)
            
        except Exception as e:
            progress_bar.progress(0)
            status_container.error(f"❌ Error during processing: {str(e)}")
            st.error("Please check your inputs and try again.")

def display_results_section():
    """Display the results section."""
    if not st.session_state.processing_complete or not st.session_state.research_results:
        st.info("👆 Please run the research process to see results here.")
        return
    
    results = st.session_state.research_results
    
    st.markdown('<h2 class="section-header">📊 Research Results</h2>', unsafe_allow_html=True)
    
    # Results overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Papers Found", len(results.get('papers', [])))
    with col2:
        st.metric("Summaries", len(results.get('summaries', [])))
    with col3:
        st.metric("Syntheses", len(results.get('syntheses', [])))
    with col4:
        st.metric("Podcast Files", len(results.get('audio_files', [])))
    
    # Tabs for different result types
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Papers", "📝 Summaries", "🔄 Syntheses", "�️ Podcasts"])
    
    with tab1:
        display_papers_tab(results.get('papers', []))
    
    with tab2:
        display_summaries_tab(results.get('summaries', []), results.get('audio_files', []))
    
    with tab3:
        display_syntheses_tab(results.get('syntheses', []), results.get('audio_files', []))
    
    with tab4:
        display_audio_tab(results.get('audio_files', []))

def display_papers_tab(papers):
    """Display the papers tab."""
    if not papers:
        st.info("No papers found.")
        return
    
    st.markdown(f"### Found {len(papers)} Papers")
    
    # Group papers by topic if available
    papers_by_topic = {}
    for paper in papers:
        topic = paper.get('topic', 'Uncategorized')
        if topic not in papers_by_topic:
            papers_by_topic[topic] = []
        papers_by_topic[topic].append(paper)
    
    for topic, topic_papers in papers_by_topic.items():
        with st.expander(f"🏷️ {topic} ({len(topic_papers)} papers)", expanded=True):
            for paper in topic_papers:
                st.markdown(f"**{paper.get('title', 'Unknown Title')}**")
                
                authors = paper.get('authors', [])
                if isinstance(authors, list):
                    authors_str = ', '.join(authors)
                else:
                    authors_str = str(authors)
                st.markdown(f"*Authors: {authors_str}*")
                
                st.markdown(f"*Source: {paper.get('source', 'Unknown')} | Published: {paper.get('published_date', 'Unknown')}*")
                
                abstract = paper.get('abstract', '')
                if abstract:
                    if len(abstract) > 300:
                        st.markdown(f"**Abstract:** {abstract[:300]}...")
                    else:
                        st.markdown(f"**Abstract:** {abstract}")
                
                url = paper.get('url', '')
                if url:
                    st.markdown(f"[📎 View Paper]({url})")
                
                st.markdown("---")

def display_summaries_tab(summaries, audio_files):
    """Display the summaries tab with enhanced comprehensive summaries."""
    if not summaries:
        st.info("No summaries generated.")
        return
    
    st.markdown(f"### Paper Summaries ({len(summaries)})")
    
    for summary in summaries:
        title = summary.get('title', 'Unknown Paper')
        
        with st.expander(f"📄 {title}", expanded=False):
            # Check if this is a comprehensive summary (new format)
            if isinstance(summary.get('summary'), dict):
                summary_data = summary['summary']
                
                # Summary tabs for different formats
                sum_tab1, sum_tab2, sum_tab3, sum_tab4 = st.tabs([
                    "📝 Main Summary", 
                    "🎯 Key Points", 
                    "👔 Executive", 
                    "🔬 Technical"
                ])
                
                with sum_tab1:
                    st.markdown("**Main Summary:**")
                    st.markdown(summary_data.get('main_summary', 'Summary not available'))
                    
                    # Metadata
                    metadata = summary_data.get('metadata', {})
                    if metadata:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Source", metadata.get('text_source', 'N/A'))
                        with col2:
                            compression = metadata.get('compression_ratio', 0)
                            if compression > 0:
                                st.metric("Compression", f"{compression:.1f}x")
                            else:
                                st.metric("Compression", 'N/A')
                        with col3:
                            length = metadata.get('summary_length', 0)
                            st.metric("Summary Length", f"{length} chars" if length else 'N/A')
                
                with sum_tab2:
                    st.markdown("**Key Points:**")
                    key_points = summary_data.get('key_points', [])
                    if key_points:
                        for i, point in enumerate(key_points, 1):
                            st.markdown(f"{i}. {point}")
                    else:
                        st.info("No key points extracted.")
                
                with sum_tab3:
                    st.markdown("**Executive Summary:**")
                    exec_summary = summary_data.get('executive_summary', '')
                    if exec_summary:
                        st.markdown(exec_summary)
                    else:
                        st.info("Executive summary not available.")
                
                with sum_tab4:
                    st.markdown("**Technical Summary:**")
                    tech_summary = summary_data.get('technical_summary', '')
                    if tech_summary:
                        st.markdown(tech_summary)
                    else:
                        st.info("Technical summary not available.")
                        
            else:
                # Legacy format - simple summary
                st.markdown("**Summary:**")
                st.markdown(summary.get('summary', 'Summary not available'))
            
            # Authors and metadata
            st.markdown("---")
            authors = summary.get('authors', [])
            if authors and authors != ["Unknown Author"]:
                authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
                st.markdown(f"**Authors:** {authors_str}")
            
            # Source information
            source = summary.get('source', '')
            if source:
                st.markdown(f"**Source:** {source}")
            
            # Find corresponding podcast file
            paper_id = summary.get('paper_id', '')
            podcast_file = next((af for af in audio_files if af.get('content_id') == paper_id), None)
            
            if podcast_file and os.path.exists(podcast_file.get('file_path', '')):
                st.markdown("---")
                
                # Check if it's a podcast or regular audio
                content_type = podcast_file.get('content_type', 'audio')
                if content_type in ['podcast', 'synthesis_podcast']:
                    st.markdown("**🎙️ Podcast Version:**")
                    st.markdown("*Conversational discussion between Host, Learner, and Expert*")
                    
                    # Show script preview if available
                    if 'script' in podcast_file:
                        with st.expander("📝 View Podcast Script"):
                            st.text_area("Script Preview", podcast_file['script'][:1000] + "..." if len(podcast_file['script']) > 1000 else podcast_file['script'], height=200)
                else:
                    st.markdown("**🎵 Audio Version:**")
                
                try:
                    audio_file_path = podcast_file['file_path']
                    with open(audio_file_path, 'rb') as audio_file_obj:
                        st.audio(audio_file_obj.read(), format='audio/mp3')
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")
            
            # Citation
            citation = summary.get('citation', '')
            if citation:
                st.markdown("---")
                st.markdown("**📖 Citation:**")
                st.code(citation, language="text")

def display_syntheses_tab(syntheses, audio_files):
    """Display the syntheses tab."""
    if not syntheses:
        st.info("No syntheses generated. Syntheses require multiple papers on the same topic.")
        return
    
    st.markdown(f"### Topic Syntheses ({len(syntheses)})")
    
    for synthesis in syntheses:
        topic = synthesis.get('topic', 'Unknown Topic')
        paper_count = synthesis.get('paper_count', 0)
        
        with st.expander(f"🔄 Synthesis: {topic} ({paper_count} papers)", expanded=False):
            st.markdown("**Synthesis:**")
            st.markdown(synthesis.get('synthesis', 'Synthesis not available'))
            
            # Find corresponding audio file
            synthesis_id = f"synthesis_{topic}"
            audio_file = next((af for af in audio_files if af.get('content_id') == synthesis_id), None)
            
            if audio_file and os.path.exists(audio_file.get('file_path', '')):
                st.markdown("**🎵 Audio Version:**")
                try:
                    audio_file_path = audio_file['file_path']
                    with open(audio_file_path, 'rb') as audio_file_obj:
                        st.audio(audio_file_obj.read(), format='audio/mp3')
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")
            
            # Papers included
            paper_ids = synthesis.get('paper_ids', [])
            if paper_ids:
                st.markdown("**📚 Papers included in this synthesis:**")
                for i, paper_id in enumerate(paper_ids, 1):
                    st.markdown(f"{i}. Paper ID: {paper_id}")

def display_audio_tab(audio_files):
    """Display the podcast files tab."""
    if not audio_files:
        st.info("No podcast files generated.")
        return
    
    st.markdown(f"### Podcast Files ({len(audio_files)})")
    st.info("�️ All summaries and syntheses are available as conversational podcasts with Host, Learner, and Expert personas.")
    
    # Group audio files by type
    podcast_audios = [af for af in audio_files if af.get('content_type') in ['podcast', 'summary']]
    synthesis_podcasts = [af for af in audio_files if af.get('content_type') in ['synthesis_podcast', 'synthesis']]
    
    if podcast_audios:
        st.markdown("#### 📄 Paper Summary Podcasts")
        for audio in podcast_audios:
            file_path = audio.get('file_path', '')
            content_id = audio.get('content_id', 'Unknown')
            content_type = audio.get('content_type', 'audio')
            
            if os.path.exists(file_path):
                # Show podcast information
                if content_type == 'podcast':
                    st.markdown(f"**🎙️ Podcast for:** {content_id}")
                    st.markdown("*Conversational discussion between Host, Learner, and Expert*")
                    
                    # Show script preview if available
                    if 'script' in audio:
                        with st.expander("📝 View Podcast Script"):
                            st.text_area(f"Script for {content_id}", audio['script'], height=300, key=f"script_{content_id}")
                else:
                    st.markdown(f"**🎵 Audio for:** {content_id}")
                
                try:
                    with open(file_path, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/mp3')
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")
                st.markdown("---")
    
    if synthesis_podcasts:
        st.markdown("#### 🔄 Topic Synthesis Podcasts")
        for audio in synthesis_podcasts:
            file_path = audio.get('file_path', '')
            content_id = audio.get('content_id', 'Unknown')
            content_type = audio.get('content_type', 'audio')
            
            if os.path.exists(file_path):
                # Show podcast information
                if content_type == 'synthesis_podcast':
                    st.markdown(f"**🎙️ Synthesis Podcast for:** {content_id}")
                    st.markdown("*Cross-paper synthesis discussion between Host, Learner, and Expert*")
                    
                    # Show script preview if available
                    if 'script' in audio:
                        with st.expander("📝 View Synthesis Script"):
                            st.text_area(f"Synthesis Script for {content_id}", audio['script'], height=300, key=f"synthesis_script_{content_id}")
                else:
                    st.markdown(f"**🎵 Synthesis Audio for:** {content_id}")
                
                try:
                    with open(file_path, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/mp3')
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")
                st.markdown("---")
                st.markdown("---")

def display_sidebar():
    """Display the sidebar with system information and controls."""
    with st.sidebar:
        st.markdown("## 🛠️ System Information")
        
        # System status
        if st.session_state.system_initialized:
            st.success("✅ System Initialized")
        else:
            st.error("❌ System Not Initialized")
        
        # Configuration info
        st.markdown("### ⚙️ Configuration")
        st.info(f"""
        **Max Papers per Search:** {Config.ARXIV_MAX_RESULTS}
        **Audio Language:** {Config.AUDIO_LANGUAGE}
        **Data Directory:** {Config.DATA_DIR}
        """)
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.research_results = None
            st.session_state.processing_complete = False
            st.rerun()
        
        if st.button("📁 Open Data Folder", use_container_width=True):
            st.info(f"Data folder: {Config.DATA_DIR}")
        
        # Recent results
        if st.session_state.research_results:
            st.markdown("### 📊 Current Results")
            results = st.session_state.research_results
            st.metric("Papers", len(results.get('papers', [])))
            st.metric("Summaries", len(results.get('summaries', [])))
            st.metric("Syntheses", len(results.get('syntheses', [])))
            
            if st.button("💾 Download Results", use_container_width=True):
                st.info("Download functionality would be implemented here")
        
        # Help section
        st.markdown("### ❓ Help")
        with st.expander("How to use this system"):
            st.markdown("""
            1. **Enter a research query** to search arXiv
            2. **Define topics** for paper classification
            3. **Upload PDFs** or provide URLs/DOIs
            4. **Set filters** for search parameters
            5. **Click 'Start Research Process'**
            6. **View results** in the tabs above
            7. **Listen to audio summaries** for podcast experience
            """)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Check system configuration
    if not check_system_configuration():
        st.stop()
    
    # Initialize research system
    try:
        if not st.session_state.system_initialized:
            with st.spinner("Initializing system..."):
                research_system = ResearchSystem()
                st.session_state.system_initialized = True
        else:
            research_system = ResearchSystem()
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.stop()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        query, topics, uploaded_files, urls, dois, filters = display_input_section()
        
        # Processing section
        display_processing_section(research_system, query, topics, filters, uploaded_files, urls, dois)
    
    with col2:
        # System status and quick info
        st.markdown("### 📈 System Status")
        if st.session_state.processing_complete:
            st.success("✅ Processing Complete")
        else:
            st.info("⏳ Ready for Processing")
        
        # Quick stats
        if st.session_state.research_results:
            results = st.session_state.research_results
            st.markdown("#### 📊 Quick Stats")
            st.metric("Total Papers", len(results.get('papers', [])))
            st.metric("Total Audio Files", len(results.get('audio_files', [])))
    
    # Results section (full width)
    display_results_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔬 Research Paper Multi-Agent System | Built with Streamlit, OpenAI, and gTTS</p>
    <p>This system demonstrates a minimal but functional multi-agent approach to research paper analysis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
