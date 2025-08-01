"""
Text-to-Speech Service

This module provides text-to-speech capabilities using Google Text-to-Speech (gTTS).
It handles audio file generation, text processing, filename sanitization, and error management
for converting research content into audio format.

Classes:
    TTSService: Main service class for text-to-speech operations

Features:
    - Google Text-to-Speech integration
    - Text length management and truncation
    - Filesystem-safe filename generation
    - Configurable language and speed settings
    - Error handling and graceful degradation
    - Audio file management and cleanup

Dependencies:
    - gtts: Google Text-to-Speech library
    - system.config: Configuration management
    - os: File system operations
"""

import os
from gtts import gTTS
from typing import Optional
from system.config import Config


class TTSService:
    """
    Text-to-Speech Service
    
    Provides text-to-speech functionality for converting research content
    into audio format using Google Text-to-Speech.
    
    This service handles:
    - Text preprocessing and length management
    - Audio file generation with configurable settings
    - Filename sanitization for filesystem compatibility
    - Error handling and graceful degradation
    - Audio format and quality optimization
    
    Attributes:
        language (str): Language code for speech synthesis
        slow (bool): Whether to use slow speech rate
        max_text_length (int): Maximum text length for TTS processing
        
    Methods:
        generate_audio(text, filename): Main method for audio generation
        _validate_text_input(text): Validates and preprocesses text input
        _prepare_text_for_tts(text): Prepares text for TTS processing
        _create_tts_instance(text): Creates configured gTTS instance
        _save_audio_file(tts_instance, filename): Saves audio to file
        _clean_filename(filename): Sanitizes filename for filesystem safety
        _get_safe_filename_chars(filename): Extracts safe characters from filename
    """
    
    def __init__(self):
        """
        Initialize the TTS service with configuration settings.
        
        Sets up language preferences, speech rate, and processing limits
        based on system configuration.
        """
        self.language = Config.AUDIO_LANGUAGE
        self.slow = Config.AUDIO_SLOW
        self.max_text_length = 5000
    
    def generate_audio(self, text: str, filename: str) -> Optional[str]:
        """
        Generate audio file from text using Google Text-to-Speech.
        
        This method orchestrates the complete audio generation process including
        text validation, preprocessing, TTS conversion, and file saving.
        
        Args:
            text (str): Text content to convert to speech. Should be clean text
                       without excessive formatting or special characters.
            filename (str): Base filename for the audio file (without extension).
                           Will be sanitized for filesystem compatibility.
            
        Returns:
            Optional[str]: Full path to the generated audio file if successful,
                          None if generation failed.
                          
        Raises:
            Exception: Logs but does not propagate exceptions, returns None instead.
            
        Example:
            >>> tts_service = TTSService()
            >>> audio_path = tts_service.generate_audio("Hello world", "greeting")
            >>> print(audio_path)  # "/path/to/audio/greeting.mp3"
        """
        try:
            # Validate and prepare input text
            if not self._validate_text_input(text):
                return None
                
            processed_text = self._prepare_text_for_tts(text)
            
            # Create TTS instance with processed text
            tts_instance = self._create_tts_instance(processed_text)
            if not tts_instance:
                return None
                
            # Save audio file and return path
            return self._save_audio_file(tts_instance, filename)
            
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None
    
    def _validate_text_input(self, text: str) -> bool:
        """
        Validate text input for TTS processing.
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if text is valid for processing, False otherwise
        """
        if not text or not isinstance(text, str):
            print("Error: Invalid text input for audio generation")
            return False
            
        if not text.strip():
            print("Error: Empty text provided for audio generation")
            return False
            
        return True
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """
        Prepare text for TTS processing by applying length limits and cleanup.
        
        Args:
            text (str): Original text content
            
        Returns:
            str: Processed text ready for TTS conversion
        """
        # Strip whitespace and normalize
        processed_text = text.strip()
        
        # Apply length limits to prevent TTS errors
        if len(processed_text) > self.max_text_length:
            processed_text = processed_text[:self.max_text_length]
            processed_text += "... Content truncated for audio generation."
            
        return processed_text
    
    def _create_tts_instance(self, text: str) -> Optional[gTTS]:
        """
        Create a configured gTTS instance for text-to-speech conversion.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            Optional[gTTS]: Configured gTTS instance or None if creation failed
        """
        try:
            return gTTS(text=text, lang=self.language, slow=self.slow)
        except Exception as e:
            print(f"Error creating TTS instance: {str(e)}")
            return None
    
    def _save_audio_file(self, tts_instance: gTTS, filename: str) -> Optional[str]:
        """
        Save TTS audio to file with proper path handling.
        
        Args:
            tts_instance (gTTS): Configured TTS instance
            filename (str): Base filename for the audio file
            
        Returns:
            Optional[str]: Full path to saved audio file or None if failed
        """
        try:
            # Clean filename for filesystem safety
            clean_filename = self._clean_filename(filename)
            
            # Generate full path
            audio_path = os.path.join(Config.AUDIO_DIR, f"{clean_filename}.mp3")
            
            # Ensure directory exists
            os.makedirs(Config.AUDIO_DIR, exist_ok=True)
            
            # Save audio file
            tts_instance.save(audio_path)
            
            print(f"Audio generated successfully: {audio_path}")
            return audio_path
            
        except Exception as e:
            print(f"Error saving audio file: {str(e)}")
            return None
    
    def _clean_filename(self, filename: str) -> str:
        """
        Clean filename to make it filesystem-safe and compatible across platforms.
        
        This method removes invalid characters, replaces spaces with underscores,
        applies length limits, and ensures the filename is never empty.
        
        Args:
            filename (str): Original filename that may contain invalid characters
            
        Returns:
            str: Sanitized filename safe for use across different filesystems
            
        Example:
            >>> service = TTSService()
            >>> clean = service._clean_filename("My Research: Paper #1")
            >>> print(clean)  # "My_Research__Paper__1"
        """
        if not filename:
            return "audio_file"
            
        # Apply character sanitization
        sanitized = self._get_safe_filename_chars(filename)
        
        # Apply length limits
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        # Ensure not empty after processing
        return sanitized if sanitized else "audio_file"
    
    def _get_safe_filename_chars(self, filename: str) -> str:
        """
        Extract safe characters from filename and replace invalid ones.
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Filename with only safe characters
        """
        # Define invalid characters for filesystem compatibility
        invalid_chars = '<>:"/\\|?*'
        
        # Replace invalid characters with underscores
        sanitized = filename
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Replace spaces with underscores for consistency
        sanitized = sanitized.replace(' ', '_')
        
        # Remove multiple consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Remove leading/trailing underscores
        return sanitized.strip('_')
