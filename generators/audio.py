"""Audio generation for podcast content using the Kokoro TTS models."""

import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:/Program Files/eSpeak NG/libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:/Program Files/eSpeak NG/espeak-ng.exe"

import torch
import numpy as np
import logging
import soundfile as sf
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from ..core.podcast import Podcast
from ..core.config import PodcastConfig
from ..utils.exceptions import AudioGenerationError

logger = logging.getLogger(__name__)


class AudioGenerator:
    """Generates audio for podcast content using Kokoro TTS models."""
    
    # Valid language codes and their descriptions
    LANG_CODES = {
        'a': 'American English',
        'b': 'British English',
        'j': 'Japanese',
        'h': 'Hindi',
        'p': 'Portuguese',
        'z': 'Chinese (Mandarin)',
        'i': 'Italian',
        'f': 'French',
        'e': 'Spanish'
    }
    
    # Valid voice prefixes for each language
    VOICE_PREFIXES = {
        'a': ['af_', 'am_'],  # American English female, male
        'b': ['bf_', 'bm_'],  # British English female, male
        'j': ['jf_', 'jm_'],  # Japanese female, male
        'h': ['hf_', 'hm_'],  # Hindi female, male
        'p': ['pf_', 'pm_'],  # Portuguese female, male
        'z': ['zf_', 'zm_'],  # Chinese female, male
        'i': ['if_', 'im_'],  # Italian female, male
        'f': ['ff_'],         # French female
        'e': ['ef_', 'em_']   # Spanish female, male
    }
    
    def __init__(self, config: PodcastConfig):
        """Initialize the audio generator with Kokoro KPipeline."""
        self.config = config
        self.output_dir = Path(config.audio.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_size = config.audio.chunk_size
        self.lang_code = config.audio.lang
        
        # Validate language code
        if self.lang_code not in self.LANG_CODES:
            valid_codes = ', '.join(f"'{code}' ({desc})" for code, desc in self.LANG_CODES.items())
            raise ValueError(f"Invalid language code '{self.lang_code}'. Valid codes are: {valid_codes}")
        
        # Extract voice names - use original field names (host_audio, etc.)
        self.host_voice = self._extract_voice_name(config.audio.host_audio)
        self.moderator_voice = self._extract_voice_name(config.audio.moderator_audio)
        self.guest_voice = self._extract_voice_name(config.audio.guest_audio)
        
        # Validate voice names match language code
        self._validate_voice(self.host_voice, "host")
        self._validate_voice(self.moderator_voice, "moderator")
        self._validate_voice(self.guest_voice, "guest")
        
        # Initialize Kokoro TTS
        try:
            from kokoro import KPipeline
            
            # Initialize device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Initialize the KPipeline with appropriate language code
            logger.info(f"Initializing Kokoro KPipeline with language: {self.lang_code} ({self.LANG_CODES[self.lang_code]})")
            self.pipeline = KPipeline(lang_code=self.lang_code)
            
            # Store voice mappings
            self.voices = {
                'Host': self.host_voice,
                'Moderator': self.moderator_voice,
                'Guest': self.guest_voice
            }
            
            logger.info(f"Audio generator initialized with voices: {self.voices}")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            
            # Provide helpful installation instructions based on language
            if self.lang_code == 'j':
                extra_info = "For Japanese support, install: pip install misaki[ja]"
            elif self.lang_code == 'z':
                extra_info = "For Chinese support, install: pip install misaki[zh]"
            else:
                extra_info = "Make sure Kokoro is installed: pip install kokoro"
                
            raise AudioGenerationError(f"Failed to initialize audio generator: {str(e)}. {extra_info}")
    
    def _extract_voice_name(self, voice_path: str) -> str:
        """Extract voice name from path or use as-is if it's already a name."""
        # If the path ends with .pt, extract the basename without extension
        if voice_path.endswith('.pt'):
            return Path(voice_path).stem
        # Otherwise assume it's already a voice name
        return voice_path
    
    def _validate_voice(self, voice: str, role: str) -> None:
        """Validate that voice matches the configured language code."""
        valid_prefixes = self.VOICE_PREFIXES.get(self.lang_code, [])
        
        if not any(voice.startswith(prefix) for prefix in valid_prefixes):
            prefix_examples = ', '.join(f"'{prefix}'" for prefix in valid_prefixes)
            language_name = self.LANG_CODES[self.lang_code]
            raise ValueError(
                f"The {role} voice '{voice}' does not match the language code '{self.lang_code}' ({language_name}). "
                f"For this language, voice names should start with one of: {prefix_examples}"
            )
    
    def generate(self, podcast: Podcast) -> Dict[str, Path]:
        """Generate audio for all podcast segments."""
        logger.info("Starting audio generation")
        audio_files = {}
        
        # Generate intro
        logger.info("Generating intro audio")
        intro_path = self.output_dir / '00_intro.wav'
        self._generate_audio_for_text(podcast.intro, intro_path, self.voices['Host'])
        audio_files['intro'] = intro_path
        
        # Generate topic conversations
        for i, topic in enumerate(podcast.topics, 1):
            logger.info(f"Generating audio for topic {i}: {topic.title}")
            
            # Topic announcement
            topic_path = self.output_dir / f'{i:02d}_0_topic.wav'
            self._generate_audio_for_text(f"Topic {i}: {topic.title}", topic_path, self.voices['Host'])
            audio_files[f'topic_{i}'] = topic_path
            
            # Generate exchanges
            for j, exchange in enumerate(topic.exchanges, 1):
                logger.info(f"Generating audio for {exchange.speaker}")
                exchange_path = self.output_dir / f'{i:02d}_{j}_speaker_{exchange.speaker.lower()}.wav'
                voice = self.voices.get(exchange.speaker, self.voices['Guest'])
                self._generate_audio_for_text(exchange.content, exchange_path, voice)
                audio_files[f'topic_{i}_exchange_{j}'] = exchange_path
        
        # Generate outro
        logger.info("Generating outro audio")
        outro_path = self.output_dir / '99_outro.wav'
        self._generate_audio_for_text(podcast.outro, outro_path, self.voices['Host'])
        audio_files['outro'] = outro_path
        
        logger.info(f"Audio generation complete. Generated {len(audio_files)} audio files.")
        return audio_files
    
    def _generate_audio_for_text(self, text: str, output_path: Path, voice: str) -> None:
        """Generate audio for text using Kokoro KPipeline."""
        try:
            logger.info(f"Generating audio for text with voice {voice}")
            
            # Use KPipeline's built-in text splitting
            generator = self.pipeline(
                text, 
                voice=voice, 
                speed=1.0, 
                split_pattern=r'\n+'
            )
            
            audio_chunks = []
            
            # Process each chunk returned by the generator
            generator_items = list(generator)  # Convert to list to get length for progress bar
            
            for i, (graphemes, phonemes, audio) in enumerate(tqdm(generator_items, desc="Generating audio")):
                audio_chunks.append(audio)
                
                # Add a small pause between paragraphs for natural speech
                if i < len(generator_items) - 1:
                    pause = np.zeros(int(24000 * 0.3))  # 0.3 second pause
                    audio_chunks.append(pause)
            
            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)
                sf.write(output_path, final_audio, 24000)
                logger.info(f"Audio saved to {output_path}")
            else:
                logger.warning(f"No audio generated for {output_path}")
                
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            logger.error(f"Failed text: {text[:100]}...")
            
            # Create a helpful error message
            error_msg = f"Failed to generate audio with voice '{voice}' for language '{self.lang_code}'"
            
            # Add specific language-related advice
            if "voice not found" in str(e).lower() or "voice unavailable" in str(e).lower():
                error_msg += f". Make sure the voice '{voice}' is available for {self.LANG_CODES.get(self.lang_code, 'the selected language')}"
                
                # Suggest installation commands for common languages
                if self.lang_code == 'j' and "not found" in str(e).lower():
                    error_msg += ". For Japanese support, run: pip install misaki[ja]"
                elif self.lang_code == 'z' and "not found" in str(e).lower():
                    error_msg += ". For Chinese support, run: pip install misaki[zh]"
            
            raise AudioGenerationError(f"{error_msg}: {str(e)}")