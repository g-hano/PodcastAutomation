"""Configuration management for the podcast generator."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional, Literal
from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    verbose: bool = False  # Enable verbose output of LLM conversations


@dataclass
class LLMProviderConfig:
    """Configuration for LLM API providers."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    topic_generator: str = "qwen2.5:14b"
    podcast_moderator: str = "llama3.1:8b"
    podcast_host: str = "llama3.1:8b"
    podcast_guest: str = "llama3.1:8b"
    intro_generator: str = "qwen2.5:14b"
    outro_generator: str = "llama3.1:8b"
    translator: str = "qwen2.5:14b"
    kokoro_path: Optional[str] = None  # Now optional as KPipeline doesn't require explicit model loading
    providers: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    
    def get_provider_for_model(self, model_name: str) -> tuple[str, str]:
        """Determine provider and model name based on the model string."""
        if model_name.startswith("ollama/"):
            return "ollama", model_name.split("/", 1)[1]
        elif model_name.startswith("openai/"):
            return "openai", model_name.split("/", 1)[1]
        elif model_name.startswith("anthropic/"):
            return "anthropic", model_name.split("/", 1)[1]
        elif model_name.startswith("groq/"):
            return "groq", model_name.split("/", 1)[1]
        else:
            # Default to ollama for backward compatibility
            return "ollama", model_name


@dataclass
class AudioConfig:
    """Configuration for audio generation and processing."""
    lang: str = "b"  # "a" - american, "b" - british, "j" - japanese, "z" - mandarin, etc.
    # Voice configuration fields
    host_voice: str = "bf_emma"  # Voice name or path to voice model
    moderator_voice: str = "bf_isabella"
    guest_voice: str = "bm_george"
    
    # Audio configuration
    output_dir: str = "output"
    output_file: str = "podcast.wav"
    chunk_size: int = 200
    music_path: str = "music/background.mp3"
    vocal_volume: int = 0  # in dB
    bg_intro_volume: int = -12  # optional
    bg_content_volume: int = -20  # optional
    bg_outro_volume: int = -12  # optional
    
    # Subtitles and export
    generate_subtitles: bool = False  # Whether to generate subtitle files
    subtitle_format: Literal["srt", "vtt"] = "srt"  # Subtitle format
    
    # For backward compatibility (when host_audio is used)
    @property
    def host_audio(self) -> str:
        """Get host voice for backward compatibility."""
        return self.host_voice
    
    @property
    def moderator_audio(self) -> str:
        """Get moderator voice for backward compatibility."""
        return self.moderator_voice
    
    @property
    def guest_audio(self) -> str:
        """Get guest voice for backward compatibility."""
        return self.guest_voice


@dataclass
class PodcastConfig:
    """Main configuration class for podcast generation."""
    pdf_path: str = ""
    num_topics: int = 5
    num_turns: int = 6
    models: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = "output"
    export_conversation_json: bool = True  # Export detailed conversation data with timestamps
    
    @classmethod
    def from_source(cls, source: Union[Dict[str, Any], str, Path]) -> 'PodcastConfig':
        """Create a config object from dictionary, yaml file path, or YAML string."""
        if isinstance(source, dict):
            # Process nested dictionaries
            config_dict = source.copy()
            
            # Process models config with providers
            if "models" in config_dict and isinstance(config_dict["models"], dict):
                models_dict = config_dict["models"].copy()
                
                # Extract providers config if present
                providers_dict = models_dict.pop("providers", {}) if "providers" in models_dict else {}
                models_config = ModelConfig(**models_dict)
                
                # Set providers if specified
                if providers_dict:
                    models_config.providers = LLMProviderConfig(**providers_dict)
                
                config_dict["models"] = models_config
            
            # Process audio config
            if "audio" in config_dict and isinstance(config_dict["audio"], dict):
                audio_dict = config_dict["audio"].copy()
                # If old field names exist, map them to new names
                if "host_audio" in audio_dict and "host_voice" not in audio_dict:
                    audio_dict["host_voice"] = audio_dict.pop("host_audio")
                if "moderator_audio" in audio_dict and "moderator_voice" not in audio_dict:
                    audio_dict["moderator_voice"] = audio_dict.pop("moderator_audio")
                if "guest_audio" in audio_dict and "guest_voice" not in audio_dict:
                    audio_dict["guest_voice"] = audio_dict.pop("guest_audio")
                config_dict["audio"] = AudioConfig(**audio_dict)
            
            # Process logging config
            if "logging" in config_dict and isinstance(config_dict["logging"], dict):
                config_dict["logging"] = LoggingConfig(**config_dict["logging"])
            
            return cls(**config_dict)
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.is_file():
                with open(path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                return cls.from_source(config_dict)
            
            # Assume it's a YAML string
            try:
                config_dict = yaml.safe_load(source)
                return cls.from_source(config_dict)
            except yaml.YAMLError:
                raise ValueError(f"Invalid YAML source: {source}")
        
        raise TypeError(f"Unsupported config source type: {type(source)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            "pdf_path": self.pdf_path,
            "num_topics": self.num_topics,
            "num_turns": self.num_turns,
            "export_conversation_json": self.export_conversation_json,
            "models": {
                "topic_generator": self.models.topic_generator,
                "podcast_moderator": self.models.podcast_moderator,
                "podcast_host": self.models.podcast_host,
                "intro_generator": self.models.intro_generator,
                "outro_generator": self.models.outro_generator,
                "translator": self.models.translator,
                "kokoro_path": self.models.kokoro_path,
                "providers": {
                    "openai_api_key": self.models.providers.openai_api_key,
                    "anthropic_api_key": self.models.providers.anthropic_api_key,
                    "groq_api_key": self.models.providers.groq_api_key,
                    "ollama_base_url": self.models.providers.ollama_base_url
                }
            },
            "audio": {
                "lang": self.audio.lang,
                "host_voice": self.audio.host_voice,
                "moderator_voice": self.audio.moderator_voice,
                "guest_voice": self.audio.guest_voice,
                "output_dir": self.audio.output_dir,
                "output_file": self.audio.output_file,
                "chunk_size": self.audio.chunk_size,
                "music_path": self.audio.music_path,
                "vocal_volume": self.audio.vocal_volume,
                "bg_intro_volume": self.audio.bg_intro_volume,
                "bg_content_volume": self.audio.bg_content_volume,
                "bg_outro_volume": self.audio.bg_outro_volume,
                "generate_subtitles": self.audio.generate_subtitles,
                "subtitle_format": self.audio.subtitle_format
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "verbose": self.logging.verbose
            },
            "output_dir": self.output_dir
        }