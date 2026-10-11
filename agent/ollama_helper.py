import logging
from typing import Optional

class OllamaHelper:
    def __init__(self, model: str = "qwen2.5-coder:0.5b", temperature: float = 0.7, max_tokens: int = 150):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available = None
        self.logger = logging.getLogger("koks.ollama")
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import ollama
            ollama.list()
            self._available = True
            return True
        except Exception as e:
            self.logger.warning(f"Ollama не доступен: {e}")
            self._available = False
            return False
    
    def generate(self, prompt: str) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            return response.message.content.strip()
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            return None
    
    def generate_with_context(self, user_input: str, context: str = "", knowledge: str = "") -> Optional[str]:
        prompt = f"""Ты помощник. Ответь на вопрос пользователя.
        
Контекст: {context}
Знания: {knowledge}

Вопрос: {user_input}

Ответ:"""
        return self.generate(prompt)