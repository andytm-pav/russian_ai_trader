"""
Провайдер для локальной Ollama
"""
import requests
from .base_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Провайдер для Ollama API"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.url = config.get("url", "http://localhost:11434")
        self.model = config.get("model", "gemma3:1b")
        self.temperature = config.get("temperature", 0.1)

    def generate(self, prompt: str) -> str:
        """Отправляет запрос в Ollama"""
        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9
                }
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["response"]

    def is_available(self) -> bool:
        """Проверяет доступность Ollama"""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    def list_models(self) -> list:
        """Возвращает список загруженных моделей"""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []