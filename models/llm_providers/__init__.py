"""
Фабрика провайдеров LLM
"""
from .base_provider import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider

_PROVIDER_REGISTRY = {
    "ollama": OllamaProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def create_provider(config: dict) -> BaseLLMProvider:
    """Создаёт провайдера по типу из конфига"""
    provider_type = config.get("type", "ollama")
    provider_class = _PROVIDER_REGISTRY.get(provider_type)
    if provider_class is None:
        raise ValueError(f"Неизвестный тип провайдера: {provider_type}. Доступны: {list(_PROVIDER_REGISTRY.keys())}")
    return provider_class(config)