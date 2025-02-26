"""Custom exceptions for the podcast generator."""


class PodcastGenerationError(Exception):
    """Base exception for all podcast generation errors."""
    pass


class ConfigurationError(PodcastGenerationError):
    """Exception raised for configuration errors."""
    pass


class ContentGenerationError(PodcastGenerationError):
    """Exception raised for errors during content generation."""
    pass


class TranslationError(PodcastGenerationError):
    """Exception raised for errors during translation."""
    pass


class AudioGenerationError(PodcastGenerationError):
    """Exception raised for errors during audio generation."""
    pass


class AudioAssemblyError(PodcastGenerationError):
    """Exception raised for errors during audio assembly."""
    pass