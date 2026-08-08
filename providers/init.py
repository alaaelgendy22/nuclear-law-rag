from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider
from .ollama_provider import OllamaProvider
import config

def get_provider() -> LLMProvider:
    provider_name = config.LLM_PROVIDER.lower()
    if provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    elif provider_name == "openrouter":
        return OpenRouterProvider()
    elif provider_name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
