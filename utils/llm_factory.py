"""LLM factory for creating different LLM instances based on provider."""

import os
import logging
from typing import Optional, Dict, Any

from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq

from ..core.config import ModelConfig, LLMProviderConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(model_name: str, model_config: ModelConfig, 
                  context_window: int = 4096,
                  additional_kwargs: Optional[Dict[str, Any]] = None) -> LLM:
        """Create an LLM instance based on the provider and model name.
        
        Args:
            model_name: Model name, possibly with provider prefix (e.g. 'ollama/llama3')
            model_config: Model configuration
            context_window: Context window size
            additional_kwargs: Additional kwargs to pass to the LLM constructor
            
        Returns:
            LLM instance
        """
        provider, base_model_name = model_config.get_provider_for_model(model_name)
        provider_config = model_config.providers
        kwargs = additional_kwargs or {}
        
        # Set up environment variables for API keys if they aren't already set
        LLMFactory._setup_environment_variables(provider, provider_config)
        
        # Create the appropriate LLM based on provider
        if provider == "ollama":
            return Ollama(
                model=base_model_name,
                context_window=context_window,
                base_url=provider_config.ollama_base_url,
                **kwargs
            )
        elif provider == "openai":
            return OpenAI(
                model=base_model_name,
                context_window=context_window,
                api_key=provider_config.openai_api_key,
                **kwargs
            )
        elif provider == "anthropic":
            return Anthropic(
                model=base_model_name,
                context_window=context_window,
                api_key=provider_config.anthropic_api_key,
                **kwargs
            )
        elif provider == "groq":
            return Groq(
                model=base_model_name,
                context_window=context_window,
                api_key=provider_config.groq_api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _setup_environment_variables(provider: str, provider_config: LLMProviderConfig) -> None:
        """Set up environment variables for API keys."""
        if provider == "openai" and provider_config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = provider_config.openai_api_key
        elif provider == "anthropic" and provider_config.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = provider_config.anthropic_api_key
        elif provider == "groq" and provider_config.groq_api_key:
            os.environ["GROQ_API_KEY"] = provider_config.groq_api_key