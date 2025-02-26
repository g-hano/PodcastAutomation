"""Audio assembler for combining podcast segments into final output."""

import numpy as np
import soundfile as sf
import librosa
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from ..core.config import PodcastConfig
from ..utils.exceptions import AudioAssemblyError

logger = logging.getLogger(__name__)


class AudioAssembler:
    """Assembles final podcast audio from individual segments."""
    
    def __init__(self, config: PodcastConfig):
        """Initialize the audio assembler."""
        self.config = config
        self.audio_dir = Path(config.output_dir)
        self.output_file = config.audio.output_file
        self.music_path = Path(config.audio.music_path)
        self.sample_rate = 24000  # Fixed sample rate for now
        
        # Volume configuration
        self.vocal_volume = config.audio.vocal_volume
        self.bg_intro_volume = config.audio.bg_intro_volume
        self.bg_content_volume = config.audio.bg_content_volume
        self.bg_outro_volume = config.audio.bg_outro_volume
    
    def assemble(self, audio_files: Optional[Union[Dict[str, Path], List[Path]]] = None, 
                output_dir: Optional[Path] = None) -> Path:
        """Assemble all audio files into a single podcast."""
        try:
            if audio_files is None:
                # Gather all WAV files in the audio directory
                audio_files = sorted(self.audio_dir.glob('*.wav'))
                if not audio_files:
                    raise ValueError(f"No WAV files found in {self.audio_dir}")
            elif isinstance(audio_files, dict):
                # Convert to a sorted list
                audio_files = sorted(audio_files.values(), key=lambda x: x.name)
            
            output_dir = Path(output_dir or self.audio_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / self.output_file
            
            logger.info(f"Assembling {len(audio_files)} audio segments into final podcast")
            
            # Load and process background music with resampling
            logger.info(f"Loading background music from {self.music_path}")
            music_audio, _ = librosa.load(self.music_path, sr=self.sample_rate)
            
            # Extract segments for intro/outro
            samples_15s = 15 * self.sample_rate
            first_15s = music_audio[:samples_15s] if len(music_audio) >= samples_15s else music_audio
            last_15s = music_audio[-samples_15s:] if len(music_audio) >= samples_15s else music_audio
            
            # Apply volume adjustment to music segments
            if self.bg_intro_volume is not None:
                first_15s = self._adjust_volume(first_15s, self.bg_intro_volume)
            if self.bg_outro_volume is not None:
                last_15s = self._adjust_volume(last_15s, self.bg_outro_volume)
            
            podcast_segments = [first_15s]
            
            logger.info("Processing audio segments")
            for i, file in enumerate(tqdm(audio_files, desc="Assembling podcast")):
                audio, sr = sf.read(file)
                if sr != self.sample_rate:
                    logger.warning(f"File {file} has incorrect sample rate {sr}, resampling")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                # Apply volume adjustment to speech
                if self.vocal_volume is not None:
                    audio = self._adjust_volume(audio, self.vocal_volume)
                
                # Apply fade effects
                if i == 0:
                    audio = self._fade_in(audio)
                elif i == len(audio_files) - 1:
                    audio = self._fade_out(audio)
                
                podcast_segments.append(audio)
                
                # Add appropriate silence between segments
                if i != len(audio_files) - 1:  # Skip silence after last file
                    if '_topic' in file.name:
                        podcast_segments.append(self._create_silence(1.5))
                    elif '_speaker' in file.name:
                        podcast_segments.append(self._create_silence(0.7))
                    else:
                        podcast_segments.append(self._create_silence(1.0))
            
            podcast_segments.append(last_15s)
            
            final_podcast = np.concatenate(podcast_segments)
            sf.write(output_path, final_podcast, self.sample_rate)
            
            duration_minutes = len(final_podcast) / self.sample_rate / 60
            logger.info(f"Podcast assembly complete! Duration: {duration_minutes:.2f} minutes")
            logger.info(f"Final podcast saved to: {output_path.absolute()}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to assemble podcast: {str(e)}", exc_info=True)
            raise AudioAssemblyError(f"Podcast assembly failed: {str(e)}")
    
    def _create_silence(self, duration_seconds: float) -> np.ndarray:
        """Create silence of specified duration."""
        return np.zeros(int(self.sample_rate * duration_seconds))
    
    def _fade_in(self, audio: np.ndarray, duration_seconds: float = 0.5) -> np.ndarray:
        """Apply fade-in effect."""
        fade_length = int(self.sample_rate * duration_seconds)
        fade = np.linspace(0, 1, fade_length)
        audio_copy = audio.copy()  # Create a copy to avoid modifying the original
        audio_copy[:fade_length] *= fade
        return audio_copy
    
    def _fade_out(self, audio: np.ndarray, duration_seconds: float = 0.5) -> np.ndarray:
        """Apply fade-out effect."""
        fade_length = int(self.sample_rate * duration_seconds)
        fade = np.linspace(1, 0, fade_length)
        audio_copy = audio.copy()  # Create a copy to avoid modifying the original
        audio_copy[-fade_length:] *= fade
        return audio_copy
    
    def _adjust_volume(self, audio: np.ndarray, db_change: float) -> np.ndarray:
        """Adjust the volume of audio by specified dB."""
        if db_change == 0:
            return audio
        
        # Convert dB to amplitude factor
        factor = 10 ** (db_change / 20)
        return audio * factor