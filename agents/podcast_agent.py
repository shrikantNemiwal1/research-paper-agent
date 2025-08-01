"""
Podcast Generation Agent

This module provides conversational podcast generation capabilities for research content,
creating engaging dialogues between multiple personas (Host, Learner, Expert) from
research papers and synthesis content.

Classes:
    PodcastAgent: Main agent class for podcast script and audio generation

Features:
    - Multi-persona conversational script generation
    - Research paper to dialogue conversion
    - Audio file generation with FFmpeg integration
    - Section-based podcast planning and organization
    - Script enhancement and dialogue optimization
    - Audio directory management and file handling
    - Error handling and graceful degradation

Dependencies:
    - services.llm_service: Language model service for script generation
    - os: File system operations and environment management
    - tempfile: Temporary file handling for audio processing
    - typing: Type hints and annotations
"""

import os
import re
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
from services.llm_service import LLMService


class PodcastAgent:
    """
    Podcast Generation Agent
    
    Agent for generating conversational podcast scripts and audio from research content.
    Creates engaging dialogues between multiple personas (Host, Learner, Expert) and
    converts them into audio format using text-to-speech and audio processing.
    
    This agent provides:
    - Conversational script generation with multiple personas
    - Research paper to dialogue conversion
    - Section-based podcast planning and organization
    - Audio file generation with FFmpeg integration
    - Script enhancement and dialogue optimization
    - Audio directory management and file handling
    - Error handling and graceful degradation
    
    Attributes:
        llm_service (LLMService): Language model service for script generation
        audio_dir (str): Directory path for generated audio files
        
    Methods:
        generate_podcast_script(paper): Generate complete podcast script from paper
        generate_audio_from_script(script, title): Convert script to audio file
        
    Private Methods:
        _setup_audio_directory(): Initialize audio output directory
        _ensure_ffmpeg_available(): Verify FFmpeg installation
        _extract_paper_content(paper): Extract content from paper object
        _generate_plan(content): Create podcast section plan
        _generate_introduction(paper): Create podcast introduction
        _generate_section_dialogue(section, context, content): Generate section dialogue
        _enhance_script(script): Enhance and optimize script
        _create_audio_segments(script): Create individual audio segments
        _combine_audio_files(segments, output_path): Combine segments into final audio
        _clean_text_for_speech(text): Optimize text for speech synthesis
        _validate_script_content(script): Validate script before processing
    """
    def __init__(self, llm_service: LLMService):
        """
        Initialize the podcast generation agent.
        
        Sets up the language model service, audio directory, and ensures
        FFmpeg is available for audio processing operations.
        
        Args:
            llm_service (LLMService): Language model service for script generation
        """
        self.llm_service = llm_service
        self._setup_audio_directory()
        self._ensure_ffmpeg_available()
        
    def _setup_audio_directory(self):
        """
        Create and initialize audio output directory.
        
        Sets up the directory structure for storing generated podcast
        audio files with proper error handling.
        """
        from system.config import Config
        self.audio_dir = Config.AUDIO_DIR
        os.makedirs(self.audio_dir, exist_ok=True)
        
    def _ensure_ffmpeg_available(self):
        """
        Ensure FFmpeg is available for audio processing operations.
        
        Adds FFmpeg to system PATH if not already present, enabling
        audio file manipulation and conversion capabilities.
        """
        ffmpeg_path = f"C:\\Users\\{os.environ.get('USERNAME', 'User')}\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin"
        if ffmpeg_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] += os.pathsep + ffmpeg_path

    def generate_podcast_script(self, paper: Dict[str, Any]) -> str:
        """
        Generate a complete conversational podcast script from a research paper.
        
        Creates an engaging dialogue between Host, Learner, and Expert personas
        based on the research paper content, with proper section planning and
        conversation flow optimization.
        
        Args:
            paper (Dict[str, Any]): Paper object containing title, content, and metadata
            
        Returns:
            str: Complete podcast script with multi-persona dialogue
            
        Raises:
            No exceptions - all errors are handled gracefully with fallbacks
        """
        try:
            # Extract and validate paper content
            paper_content = self._extract_paper_content(paper)
            if not paper_content:
                return self._generate_fallback_script(paper)
            
            # Generate podcast structure plan
            plan_sections = self._generate_plan(paper_content)
            
            # Create introduction dialogue
            introduction = self._generate_introduction(paper)
            
            # Build complete script with sections
            script = introduction
            previous_dialogue = introduction
            
            for section in plan_sections:
                section_script = self._generate_section_dialogue(
                    section, previous_dialogue, paper_content
                )
                script += "\n" + section_script
                previous_dialogue = section_script
            
            # Enhance and optimize final script
            enhanced_script = self._enhance_script(script)
            return enhanced_script
            
        except Exception as e:
            return self._generate_fallback_script(paper)

    def generate_audio_from_script(self, script: str, filename: Optional[str] = None) -> Optional[str]:
        """Generate audio from podcast script and save to audio folder."""
        try:
            # Try Google Cloud TTS first, fallback to gTTS
            try:
                from google.cloud import texttospeech
                result = self._generate_with_google_tts(script, filename)
                if result:
                    return result
            except ImportError:
                pass
            except Exception as e:
                pass
                
            result = self._generate_with_gtts(script, filename)
            if result:
                return result
            else:
                return None
                
        except Exception as e:
            return None

    def _extract_paper_content(self, paper: Dict[str, Any]) -> str:
        """Extract content from paper dictionary."""
        content = paper.get('content', '')
        if not content:
            summary = paper.get('summary', {})
            if isinstance(summary, dict):
                content = summary.get('main_summary', '')
            else:
                content = str(summary)
        return content

    def _generate_plan(self, paper_content: str) -> List[str]:
        """Generate structured podcast plan."""
        plan_prompt = """Generate a structured plan for a podcast discussion about this research paper. 
        Create sections with titles and bullet points for three personas:
        - Host: Professional presenter who explains the paper engagingly
        - Learner: Curious questioner who asks insightful questions  
        - Expert: Deep insights provider with detailed analysis

        Paper: {paper_content}
        
        Format as:
        # Section 1: Title
        - bullet point 1
        - bullet point 2
        
        Plan:"""
        
        plan_response = self.llm_service.generate_text(
            plan_prompt.format(paper_content=paper_content),
            max_tokens=1000,
            temperature=0.6
        )
        return self._parse_plan_sections(plan_response)

    def _generate_introduction(self, paper: Dict[str, Any]) -> str:
        """Generate engaging podcast introduction."""
        paper_head = self._extract_paper_head(paper)
        
        intro_prompt = """Create an engaging 3-interaction podcast introduction for this research paper.
        
        Personas:
        - Host: Professional, warm, enthusiastic presenter
        - Learner: Curious, funny questioner
        - Expert: Provides profound insights
        
        Paper info: {paper_head}
        
        IMPORTANT FORMATTING RULES:
        - Use simple format: "Host: [text]" (no asterisks, no bold, no markdown)
        - Do NOT use **Host:** or *Host:* - just use Host:
        - No markdown formatting like **bold** or *italic* anywhere
        - No brackets [ ] or parentheses ( ) in the speech
        - Use plain text only for natural speech
        
        Format example:
        Host: Welcome to our research podcast
        Learner: That sounds fascinating
        Expert: This is an important contribution
        
        Introduction:"""
        
        return self.llm_service.generate_text(
            intro_prompt.format(paper_head=paper_head),
            max_tokens=1500,
            temperature=0.8
        )

    def _generate_section_dialogue(self, section: str, previous_dialogue: str, paper_context: str) -> str:
        """Generate dialogue for a podcast section."""
        dialogue_prompt = """Create engaging dialogue for this podcast section. Make it interactive and enthusiastic.
        
        Section: {section}
        Previous context: {previous_context}
        Paper context: {paper_context}
        
        IMPORTANT FORMATTING RULES:
        - Use simple format: "Host: [text]" (no asterisks, no bold, no markdown)
        - Do NOT use **Host:** or *Host:* - just use Host:
        - No markdown formatting like **bold** or *italic* anywhere
        - No brackets [ ] or parentheses ( ) in the speech
        - Use plain text only for natural speech
        - Complete all sentences fully. End with proper sentence endings.
        
        Dialogue:"""
        
        dialogue = self.llm_service.generate_text(
            dialogue_prompt.format(
                section=section,
                previous_context=previous_dialogue[-1000:],
                paper_context=paper_context[:2000]
            ),
            max_tokens=2000,
            temperature=0.8
        )
        
        return self._ensure_complete_script(dialogue)

    def _enhance_script(self, script: str) -> str:
        """Enhance script by removing redundancy and improving flow."""
        enhance_prompt = """Enhance this podcast script by:
        - Removing audio effects mentions and sound descriptions
        - Reducing repetition and redundancy  
        - Improving transitions between speakers
        - Ensuring all sentences are complete
        - Removing ALL markdown formatting (no **bold**, *italic*, etc.)
        - Using only plain text format: "Speaker: text"
        - No asterisks, brackets, or special characters in speech
        
        Script: {script}
        
        Enhanced script:"""
        
        enhanced = self.llm_service.generate_text(
            enhance_prompt.format(script=script),
            max_tokens=3000,
            temperature=0.6
        )
        
        return self._ensure_complete_script(enhanced)

    def _ensure_complete_script(self, script: str) -> str:
        """Check and complete truncated scripts."""
        script = script.strip()
        
        # Check for truncation indicators
        truncation_signs = [
            script.endswith(('What', 'How', 'Why', 'The', 'This', '...', ',', 'and', 'but', 'or', 'so')),
            len(script.split()) < 50,
            not script.endswith(('.', '!', '?', '"'))
        ]
        
        if any(truncation_signs):
            completion_prompt = f"""This podcast dialogue was cut off. Complete it naturally:
            
            {script}
            
            Continue and finish properly:"""
            
            try:
                completion = self.llm_service.generate_text(
                    completion_prompt,
                    max_tokens=1500,
                    temperature=0.7
                )
                return script + " " + completion.strip()
            except:
                return script
        
        return script

    def _parse_plan_sections(self, plan_text: str) -> List[str]:
        """Parse plan text into structured sections."""
        sections = []
        current_section = []
        
        lines = plan_text.strip().splitlines()
        if lines:
            lines = lines[1:]  # Skip title line
        
        header_pattern = re.compile(r"^#+\s")
        bullet_pattern = re.compile(r"^- ")
        
        for line in lines:
            if header_pattern.match(line):
                if current_section:
                    sections.append(" ".join(current_section))
                    current_section = []
                current_section.append(line.strip())
            elif bullet_pattern.match(line):
                current_section.append(line.strip())
        
        if current_section:
            sections.append(" ".join(current_section))
        
        return sections

    def _extract_paper_head(self, paper: Dict[str, Any]) -> str:
        """Extract paper title, authors and abstract for introduction."""
        title = paper.get('title', 'Research Paper')
        authors = paper.get('authors', ['Unknown'])
        authors_str = ', '.join(authors) if isinstance(authors, list) else str(authors)
        
        abstract = paper.get('abstract', '')
        if not abstract:
            summary = paper.get('summary', {})
            if isinstance(summary, dict):
                abstract = summary.get('main_summary', '')[:500]
            else:
                abstract = str(summary)[:500]
        
        return f"Title: {title}\nAuthors: {authors_str}\nAbstract: {abstract}"

    def _generate_fallback_script(self, paper: Dict[str, Any]) -> str:
        """Generate simple fallback script when main generation fails."""
        title = paper.get('title', 'Research Paper')
        summary = paper.get('summary', {})
        
        if isinstance(summary, dict):
            main_summary = summary.get('main_summary', 'No summary available')
        else:
            main_summary = str(summary)
        
        return f"""Host: Welcome to our research podcast! Today we're discussing "{title}".

Learner: That sounds fascinating! What's the main focus of this research?

Host: {main_summary[:200]}...

Expert: This work contributes significantly to the field by addressing important research questions.

Learner: What are the key takeaways for our listeners?

Host: The research offers practical implications and opens new avenues for investigation.

Expert: Indeed, this study represents an important step forward in our understanding."""

    def _generate_with_google_tts(self, script: str, filename: Optional[str] = None) -> Optional[str]:
        """Generate audio using Google Cloud Text-to-Speech."""
        from google.cloud import texttospeech
        
        client = texttospeech.TextToSpeechClient()
        speech_segments = self._parse_speech_segments(script)
        
        if not speech_segments:
            return None
            
        # Use temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_files = []
            
            for i, (speaker, text) in enumerate(speech_segments, 1):
                if not text.strip():
                    continue
                    
                voice_config = self._get_voice_config(speaker)
                
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=voice_config["language_code"],
                    name=voice_config["name"],
                    ssml_gender=getattr(texttospeech.SsmlVoiceGender, voice_config["gender"])
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=voice_config.get("speaking_rate", 1.0),
                    pitch=voice_config.get("pitch", 0.0)
                )
                
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                
                temp_file = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
                with open(temp_file, "wb") as f:
                    f.write(response.audio_content)
                    
                audio_files.append(temp_file)
            
            return self._merge_and_save_audio(audio_files, filename)

    def _generate_with_gtts(self, script: str, filename: Optional[str] = None) -> Optional[str]:
        """Generate audio using gTTS (fallback method)."""
        try:
            from gtts import gTTS
        except ImportError as e:
            return None
        
        try:
            speech_segments = self._parse_speech_segments(script)
            
            if not speech_segments:
                return None
                
            # Use temporary directory for processing  
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_files = []
                
                for i, (speaker, text) in enumerate(speech_segments, 1):
                    if not text.strip():
                        continue
                        
                    voice_params = self._get_gtts_params(speaker)
                    
                    try:
                        tts = gTTS(
                            text=text,
                            lang=voice_params["lang"],
                            slow=voice_params["slow"],
                            tld=voice_params.get("tld", "com")
                        )
                        
                        temp_file = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
                        tts.save(temp_file)
                        
                        # Verify file was created
                        if os.path.exists(temp_file):
                            audio_files.append(temp_file)
                        
                    except Exception as e:
                        continue
                
                if audio_files:
                    result = self._merge_and_save_audio(audio_files, filename)
                    return result
                else:
                    return None
                    
        except Exception as e:
            return None

    def _parse_speech_segments(self, script: str) -> List[tuple]:
        """Parse script into (speaker, text) segments and clean formatting."""
        # Clean the script first to remove formatting artifacts
        cleaned_script = self._clean_script_formatting(script)
        
        segments = re.findall(
            r"(Host|Learner|Expert):\s*(.*?)(?=(?:Host|Learner|Expert):|$)", 
            cleaned_script, re.DOTALL | re.IGNORECASE
        )
        
        # Further clean each segment's text
        cleaned_segments = []
        for speaker, text in segments:
            cleaned_text = self._clean_speech_text(text.strip())
            if cleaned_text:
                cleaned_segments.append((speaker, cleaned_text))
        
        return cleaned_segments
    
    def _clean_script_formatting(self, script: str) -> str:
        """Remove formatting artifacts from podcast script."""
        # Remove markdown formatting
        script = re.sub(r'\*\*(.*?)\*\*', r'\1', script)  # Remove **bold**
        script = re.sub(r'\*(.*?)\*', r'\1', script)      # Remove *italic*
        script = re.sub(r'__(.*?)__', r'\1', script)      # Remove __underline__
        script = re.sub(r'_(.*?)_', r'\1', script)        # Remove _underline_
        script = re.sub(r'`(.*?)`', r'\1', script)        # Remove `code`
        script = re.sub(r'#{1,6}\s*', '', script)         # Remove ## headers
        
        # Clean up speaker labels that might have formatting
        script = re.sub(r'\*\*(Host|Learner|Expert)\*\*:', r'\1:', script)
        script = re.sub(r'\*(Host|Learner|Expert)\*:', r'\1:', script)
        
        return script
    
    def _clean_speech_text(self, text: str) -> str:
        """Clean individual speech text for natural audio."""
        # Remove any remaining markdown or formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)      # Remove **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)          # Remove *italic*
        text = re.sub(r'__(.*?)__', r'\1', text)          # Remove __underline__
        text = re.sub(r'_(.*?)_', r'\1', text)            # Remove _underline_
        text = re.sub(r'`(.*?)`', r'\1', text)            # Remove `code`
        text = re.sub(r'#{1,6}\s*', '', text)             # Remove headers
        
        # Remove brackets and parenthetical notes
        text = re.sub(r'\[.*?\]', '', text)               # Remove [notes]
        text = re.sub(r'\(.*?\)', '', text)               # Remove (parenthetical)
        
        # Clean up quotation marks that might sound weird
        text = text.replace('"', '')
        text = text.replace("'", '')
        
        # Remove bullet points and list markers
        text = re.sub(r'^[-•*]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short segments that are just noise
        if len(text) < 10:
            return ""
            
        return text

    def _get_voice_config(self, speaker: str) -> Dict[str, Any]:
        """Get Google Cloud TTS voice configuration."""
        configs = {
            "Host": {
                "language_code": "en-US", "name": "en-US-Journey-D", 
                "gender": "MALE", "speaking_rate": 1.0, "pitch": 0.0
            },
            "Learner": {
                "language_code": "en-US", "name": "en-US-Journey-F",
                "gender": "FEMALE", "speaking_rate": 1.1, "pitch": 2.0
            },
            "Expert": {
                "language_code": "en-US", "name": "en-US-Casual-K",
                "gender": "MALE", "speaking_rate": 0.9, "pitch": -2.0
            }
        }
        return configs.get(speaker, configs["Host"])

    def _get_gtts_params(self, speaker: str) -> Dict[str, Any]:
        """Get gTTS parameters for different speakers."""
        params = {
            "Host": {"lang": "en", "slow": False, "tld": "com"},
            "Learner": {"lang": "en", "slow": False, "tld": "co.uk"}, 
            "Expert": {"lang": "en", "slow": False, "tld": "com.au"}
        }
        return params.get(speaker, params["Host"])

    def _merge_and_save_audio(self, audio_files: List[str], filename: Optional[str] = None) -> Optional[str]:
        """Merge audio files and save to clean audio directory."""
        try:
            from pydub import AudioSegment
        except ImportError as e:
            # Return first file if pydub not available
            if audio_files:
                final_filename = filename or f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                final_path = os.path.join(self.audio_dir, final_filename)
                
                # Ensure audio directory exists
                os.makedirs(self.audio_dir, exist_ok=True)
                
                shutil.copy2(audio_files[0], final_path)
                return final_path
            return None
        
        if not audio_files:
            return None
            
        try:
            # Merge audio segments with pauses
            merged_audio = AudioSegment.empty()
            for i, audio_file in enumerate(audio_files):
                if os.path.exists(audio_file):
                    audio_segment = AudioSegment.from_mp3(audio_file)
                    if len(merged_audio) > 0:
                        merged_audio += AudioSegment.silent(duration=500)  # 0.5s pause
                    merged_audio += audio_segment
            
            # Save final podcast to clean audio directory
            final_filename = filename or f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            final_path = os.path.join(self.audio_dir, final_filename)
            
            # Ensure audio directory exists
            os.makedirs(self.audio_dir, exist_ok=True)
            
            merged_audio.export(final_path, format="mp3")
            
            if os.path.exists(final_path):
                return final_path
            else:
                return None
                
        except Exception as e:
            return None
