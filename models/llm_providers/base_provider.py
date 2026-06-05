"""
Абстрактный базовый класс для LLM-провайдеров
"""
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Базовый класс для всех провайдеров"""

    def __init__(self, config: dict):
        self.config = config
        self.timeout = config.get("timeout", 30)

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Отправляет промпт и возвращает ответ"""
        pass

    def is_available(self) -> bool:
        """Проверяет доступность провайдера"""
        return True