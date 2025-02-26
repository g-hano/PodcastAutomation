"""Logging configuration for podcast generator."""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..core.config import LoggingConfig


def configure_logging(config: LoggingConfig, log_file: Optional[str] = None) -> None:
    """Configure the logging system.
    
    Args:
        config: Logging configuration
        log_file: Optional log file path. If not provided, uses config.file
    """
    # Set log level
    level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Create log directory if needed
    if log_file or config.file:
        log_path = Path(log_file or config.file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create formatters
    file_formatter = logging.Formatter(config.format)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file or config.file:
        file_path = log_file or config.file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Create module loggers
    logging.getLogger('podcast_generator').setLevel(level)