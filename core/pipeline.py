"""Main podcast generation pipeline."""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

from .config import PodcastConfig
from .podcast import Podcast
from ..generators.translation import TranslationService
from ..generators.audio import AudioGenerator
from ..assembler.audio_assembler import AudioAssembler
from ..simulation import PodcastSimulation
from ..utils.exceptions import PodcastGenerationError

logger = logging.getLogger(__name__)


class PodcastPipeline:
    """Main pipeline for podcast generation."""
    
    def __init__(self, config: PodcastConfig):
        """Initialize the podcast pipeline with configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.simulation = PodcastSimulation(config)
        self.translation_service = TranslationService(config)
        self.audio_generator = AudioGenerator(config)
        self.audio_assembler = AudioAssembler(config)
        
        # Detect if translation to target language is needed
        self.translate_to_target = config.audio.lang not in ['a']  # 'a' is American English
        if self.translate_to_target:
            target_language = self.translation_service.LANGUAGE_NAMES.get(config.audio.lang, "the target language")
            logger.info(f"Content will be translated to {target_language}")
        
        # Timing metrics
        self.timing_metrics = {}
    
    def run(self, skip_stages: Optional[Dict[str, bool]] = None) -> Path:
        """Run the complete podcast generation pipeline.
        
        Args:
            skip_stages: Optional dictionary to skip specific stages.
                Keys: 'content', 'translation', 'audio', 'assembly'
                
        Returns:
            Path to the final podcast audio file
        """
        skip = skip_stages or {}
        total_start = time.time()
        podcast = None
        audio_files = None
        podcast_json_path = None
        
        try:
            # Stage 1: Generate content (always in English)
            if not skip.get('content', False):
                start = time.time()
                logger.info("Starting content generation")
                podcast = self.simulation.run_podcast_simulation()
                end = time.time()
                self.timing_metrics['content_generation'] = end - start
                logger.info(f"Content generation completed in {end - start:.2f} seconds")
                
                # Save raw content
                podcast_json_path = self._save_podcast_json(podcast, "podcast_data_original.json")
            else:
                logger.info("Skipping content generation")
                # Load from existing file
                podcast_json_path = self.output_dir / "podcast_data_original.json"
                if not podcast_json_path.exists():
                    podcast_json_path = self.output_dir / "podcast_data.json"
                
                with open(podcast_json_path, 'r', encoding='utf-8') as f:
                    podcast_data = json.load(f)
                    podcast = Podcast.from_dict(podcast_data)
            
            # Stage 2: Translation
            if not skip.get('translation', False):
                start = time.time()
                logger.info("Starting translation processing")
                podcast = self.translation_service.process(podcast)
                end = time.time()
                self.timing_metrics['translation'] = end - start
                logger.info(f"Translation completed in {end - start:.2f} seconds")
                
                # Save translated content
                json_filename = "podcast_data_translated.json"
                podcast_json_path = self._save_podcast_json(podcast, json_filename)
            
            # Stage 3: Audio generation
            if not skip.get('audio', False):
                start = time.time()
                logger.info("Starting audio generation")
                audio_files = self.audio_generator.generate(podcast)
                end = time.time()
                self.timing_metrics['audio_generation'] = end - start
                logger.info(f"Audio generation completed in {end - start:.2f} seconds")
            
            # Stage 4: Audio assembly
            if not skip.get('assembly', False):
                start = time.time()
                logger.info("Starting audio assembly")
                final_podcast_path = self.audio_assembler.assemble(audio_files)
                end = time.time()
                self.timing_metrics['audio_assembly'] = end - start
                logger.info(f"Audio assembly completed in {end - start:.2f} seconds")
            else:
                # Just return the default path
                final_podcast_path = self.output_dir / self.config.audio.output_file
            
            total_end = time.time()
            self.timing_metrics['total'] = total_end - total_start
            
            # Log timing summary
            self._log_timing_summary()
            
            return final_podcast_path
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise PodcastGenerationError(f"Pipeline execution failed: {str(e)}")
    
    def _save_podcast_json(self, podcast: Podcast, filename: str = "podcast_data.json") -> Path:
        """Save podcast data to JSON file."""
        json_path = self.output_dir / filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(podcast.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved podcast data to {json_path}")
        return json_path
    
    def _log_timing_summary(self) -> None:
        """Log a summary of timing metrics."""
        logger.info("-" * 40)
        logger.info("Timing Summary:")
        for stage, duration in self.timing_metrics.items():
            logger.info(f"  {stage.replace('_', ' ').title()}: {duration:.2f} seconds")
        logger.info("-" * 40)