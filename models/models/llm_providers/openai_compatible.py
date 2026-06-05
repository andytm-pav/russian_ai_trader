"""
Провайдер для OpenAI-совместимых API (OpenRouter, DeepSeek, LM Studio, vLLM)
"""
import json
import requests
from .base_provider import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    """Провайдер для OpenAI-совместимых API"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.1)

    def generate(self, prompt: str) -> str:
        """Отправляет запрос в OpenAI-совместимый API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": self.temperature
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def is_available(self) -> bool:
        """Проверяет доступность API (требует ключ)"""
        return bool(self.api_key)