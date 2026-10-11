import json
import re
from typing import Dict, Optional, Any

class Planner:
    def __init__(self):
        self.max_steps = 10
    
    def parse_action(self, response: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        for start in (match.start() for match in re.finditer(r'\{', response)):
            try:
                data, _ = decoder.raw_decode(response[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and ('tool' in data or 'done' in data):
                return data
        return None
    
    def create_system_prompt(self, tools_description: str) -> str:
        return f"""Ты - AI-ассистент с доступом к компьютеру и интернету.

Доступные инструменты:
{tools_description}

Правила:
1. Для выполнения действий используй JSON: {{"tool": "имя", "args": {{"arg": "value"}}}}
2. Если задача завершена: {{"done": true, "result": "ответ пользователю"}}
3. Рабочая директория: текущая папка
4. Разбивай сложные задачи на шаги
5. Если ошибка - предложи альтернативу

Формат ответа должен быть JSON для действий или обычный текст для ответа."""