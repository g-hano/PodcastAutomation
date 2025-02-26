"""Command-line interface for the podcast generator."""

import argparse
import sys
import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .core.config import PodcastConfig
from .core.pipeline import PodcastPipeline
from .utils.logging_config import configure_logging
from .utils.exceptions import PodcastGenerationError

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Podcast Generator - Create AI-powered podcasts from PDF documents"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for podcast files"
    )
    
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        help="Path to PDF file for podcast content"
    )
    
    parser.add_argument(
        "--topics", "-t",
        type=int,
        help="Number of topics to generate"
    )
    
    parser.add_argument(
        "--turns", "-n",
        type=int,
        help="Number of conversation turns per topic"
    )
    
    parser.add_argument(
        "--skip-content",
        action="store_true",
        help="Skip content generation stage"
    )
    
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation stage"
    )
    
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio generation stage"
    )
    
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="Skip audio assembly stage"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output showing the conversation generation"
    )
    
    # Add LLM API key arguments
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key for using OpenAI models"
    )
    
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key for using Anthropic models"
    )
    
    parser.add_argument(
        "--groq-api-key",
        type=str,
        help="Groq API key for using Groq models"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        help="Ollama server URL (default: http://localhost:11434)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {str(e)}")


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override configuration with command-line arguments."""
    # Create a copy to avoid modifying the original
    updated_config = config.copy()
    
    # Override with command-line arguments
    if args.pdf:
        updated_config["pdf_path"] = args.pdf
    
    if args.topics:
        updated_config["num_topics"] = args.topics
    
    if args.turns:
        updated_config["num_turns"] = args.turns
    
    if args.output_dir:
        updated_config["output_dir"] = args.output_dir
        
        # Also update audio output dir
        if "audio" not in updated_config:
            updated_config["audio"] = {}
        updated_config["audio"]["output_dir"] = args.output_dir
    
    # Add logging configuration
    if "logging" not in updated_config:
        updated_config["logging"] = {}
    
    updated_config["logging"]["level"] = args.log_level
    if args.log_file:
        updated_config["logging"]["file"] = args.log_file
    if args.verbose:
        updated_config["logging"]["verbose"] = True
    
    # Add LLM API keys
    if "models" not in updated_config:
        updated_config["models"] = {}
    
    if "providers" not in updated_config["models"]:
        updated_config["models"]["providers"] = {}
    
    # Override API keys if provided
    if args.openai_api_key:
        updated_config["models"]["providers"]["openai_api_key"] = args.openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        updated_config["models"]["providers"]["openai_api_key"] = os.environ["OPENAI_API_KEY"]
        
    if args.anthropic_api_key:
        updated_config["models"]["providers"]["anthropic_api_key"] = args.anthropic_api_key
    elif "ANTHROPIC_API_KEY" in os.environ:
        updated_config["models"]["providers"]["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]
        
    if args.groq_api_key:
        updated_config["models"]["providers"]["groq_api_key"] = args.groq_api_key
    elif "GROQ_API_KEY" in os.environ:
        updated_config["models"]["providers"]["groq_api_key"] = os.environ["GROQ_API_KEY"]
        
    if args.ollama_url:
        updated_config["models"]["providers"]["ollama_base_url"] = args.ollama_url
    
    return updated_config


def main() -> int:
    """Main entry point for the podcast generator CLI."""
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Load configuration
        config_dict = load_config(args.config)
        
        # Override with command-line arguments
        config_dict = override_config(config_dict, args)
        
        # Create configuration object
        config = PodcastConfig.from_source(config_dict)
        
        # Configure logging
        configure_logging(config.logging, args.log_file)
        
        logger.info("Starting podcast generation pipeline")
        
        # Check if model names follow the provider/model pattern, if not, assume ollama
        models_updated = False
        model_fields = [
            "topic_generator", "podcast_moderator", "podcast_host", 
            "intro_generator", "outro_generator", "translator"
        ]
        
        for field in model_fields:
            model_name = getattr(config.models, field)
            if model_name and "/" not in model_name:
                # Add ollama/ prefix for backward compatibility
                setattr(config.models, field, f"ollama/{model_name}")
                models_updated = True
        
        if models_updated:
            logger.info("Updated model names to include provider prefix for compatibility")
        
        # Create skip stages dictionary
        skip_stages = {
            "content": args.skip_content,
            "translation": args.skip_translation,
            "audio": args.skip_audio,
            "assembly": args.skip_assembly
        }
        
        # Run the pipeline
        pipeline = PodcastPipeline(config)
        final_podcast_path = pipeline.run(skip_stages)
        
        logger.info(f"Podcast generation completed successfully")
        logger.info(f"Final podcast saved to: {final_podcast_path}")
        
        return 0
    
    except PodcastGenerationError as e:
        logger.error(f"Podcast generation failed: {str(e)}")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())    