"""Translation service for podcast content that supports multiple languages."""

import re
import logging
from typing import Any

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

from ..core.podcast import Podcast
from ..core.config import PodcastConfig
from ..utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class TranslationService:
    """Handles translation of podcast content between languages."""
    
    # Language code to full name mapping
    LANGUAGE_NAMES = {
        'a': 'American English',
        'b': 'British English',
        'j': 'Japanese',
        'h': 'Hindi',
        'p': 'Portuguese',
        'z': 'Chinese',
        'i': 'Italian',
        'f': 'French',
        'e': 'Spanish'
    }
    
    # Default source language (for content generation)
    DEFAULT_SOURCE_LANGUAGE = 'American English'
    
    def __init__(self, config: PodcastConfig):
        """Initialize the translation service."""
        self.config = config
        self.model_name = config.models.translator
        self.target_lang_code = config.audio.lang
        
        # Get target language name
        self.target_language = self.LANGUAGE_NAMES.get(
            self.target_lang_code, 
            self.DEFAULT_SOURCE_LANGUAGE
        )
        
        logger.info(f"Translation service initialized with target language: {self.target_language} ({self.target_lang_code})")
        
        # Initialize LLM for translation using the LLMFactory
        llm = LLMFactory.create_llm(
            model_name=self.model_name,
            model_config=config.models,
            context_window=4096
        )
        
        self.translator = SimpleChatEngine(
            llm=llm,
            memory=ChatMemoryBuffer.from_defaults(llm=llm),
            prefix_messages=[self._get_translation_system_prompt()]
        )
    
    def _get_translation_system_prompt(self) -> ChatMessage:
        """Get the system prompt for the translator based on target language."""
        # If target is American English (default), we only need to translate non-English to English
        if self.target_lang_code == 'a':
            return ChatMessage(
                role="system",
                content="""You are a direct translator. When given text containing non-English content:
                1. Translate only the non-English parts to American English
                2. Keep existing English parts unchanged
                3. Output ONLY the final translated text, nothing else
                4. Do not include any translation notes or metadata
                5. Do not explain your translation process"""
            )
        
        # For other target languages, we translate from English to the target language
        return ChatMessage(
            role="system",
            content=f"""You are a professional translator specialized in {self.target_language}. When given text in English:
            1. Translate the entire text to {self.target_language}
            2. Maintain the same tone, formality level, and meaning
            3. Preserve formatting, paragraph breaks, and punctuation style
            4. Output ONLY the final translated text, nothing else
            5. Do not include any translation notes or metadata
            6. Do not explain your translation process
            7. For specialized terms without direct translations, use the most appropriate term in {self.target_language}"""
        )
    
    def process(self, podcast: Podcast) -> Podcast:
        """Process and translate podcast content to the target language."""
        # If target is American English, detect and translate non-English to English
        if self.target_lang_code == 'a':
            logger.info("Checking podcast content for non-English text")
            return self._process_to_english(podcast)
        
        # For other target languages, translate from English to target language
        logger.info(f"Translating podcast content from English to {self.target_language}")
        return self._process_to_target_language(podcast)
    
    def _process_to_english(self, podcast: Podcast) -> Podcast:
        """Process and translate non-English content to English."""
        translations_made = False
        
        # Check and translate intro if needed
        if podcast.intro and not self._is_english_only(podcast.intro):
            logger.info("Translating introduction to English")
            podcast.original_intro = podcast.intro
            podcast.intro = self._translate_to_english(podcast.intro)
            translations_made = True
        
        # Check and translate outro if needed
        if podcast.outro and not self._is_english_only(podcast.outro):
            logger.info("Translating outro to English")
            podcast.original_outro = podcast.outro
            podcast.outro = self._translate_to_english(podcast.outro)
            translations_made = True
        
        # Check and translate exchanges
        for topic in podcast.topics:
            for exchange in topic.exchanges:
                if not self._is_english_only(exchange.content):
                    logger.info(f"Translating content for topic '{topic.title}', speaker '{exchange.speaker}'")
                    exchange.original_content = exchange.content
                    exchange.content = self._translate_to_english(exchange.content)
                    translations_made = True
        
        if translations_made:
            logger.info("Translations to English completed")
        else:
            logger.info("No translations needed - all content already in English")
        
        return podcast
    
    def _process_to_target_language(self, podcast: Podcast) -> Podcast:
        """Translate English podcast content to the target language."""
        # Translate intro
        logger.info(f"Translating introduction to {self.target_language}")
        podcast.original_intro = podcast.intro
        podcast.intro = self._translate_to_target(podcast.intro)
        
        # Translate outro
        logger.info(f"Translating outro to {self.target_language}")
        podcast.original_outro = podcast.outro
        podcast.outro = self._translate_to_target(podcast.outro)
        
        # Translate all exchanges
        for topic in podcast.topics:
            logger.info(f"Translating content for topic '{topic.title}'")
            topic.title = self._translate_to_target(topic.title)
            
            for exchange in topic.exchanges:
                logger.info(f"Translating content for speaker '{exchange.speaker}'")
                exchange.original_content = exchange.content
                exchange.content = self._translate_to_target(exchange.content)
        
        logger.info(f"Translation to {self.target_language} completed")
        return podcast
    
    def _contains_non_english(self, text: str) -> bool:
        """Check if text contains non-English characters."""
        # This checks for common non-English character ranges
        non_english_pattern = re.compile(r'[\u0080-\u024F\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF]')
        return bool(non_english_pattern.search(text))
    
    def _is_english_only(self, text: str) -> bool:
        """Check if text is English only."""
        return not self._contains_non_english(text)
    
    def _clean_translation(self, translated_text: str) -> str:
        """Clean up translation output by removing any metadata or instructions."""
        # Remove common patterns that might appear in the translation
        patterns_to_remove = [
            r'Requirements:.*?(?=\n\n|\Z)',  # Remove requirements section
            r'---.*?(?=\n\n|\Z)',           # Remove separator lines
            r'Here\'s.*?translation:',       # Remove introductory phrases
            r'Translated text:',
            r'Translation:',
            r'\n\n.*?(?=\n\n|\Z)'          # Remove extra explanations
        ]
        
        text = translated_text
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _translate_to_english(self, text: str) -> str:
        """Translate non-English text to English, preserving existing English parts."""
        prompt = f"""Translate this text to American English, keeping existing English parts unchanged:

{text}

Provide ONLY the translated text, no explanations or notes."""
        
        translated = str(self.translator.chat(prompt))
        return self._clean_translation(translated)
    
    def _translate_to_target(self, text: str) -> str:
        """Translate text from English to the target language."""
        prompt = f"""Translate this text from English to {self.target_language}:

{text}

Provide ONLY the translated text in {self.target_language}, no explanations or notes."""
        
        translated = str(self.translator.chat(prompt))
        return self._clean_translation(translated)