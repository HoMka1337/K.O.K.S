import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

class Memory:
    def __init__(self, memory_path: Union[str, Path] = "memory.json", max_size: int = 20):
        self.memory_path = Path(memory_path)
        self.max_size = max_size
        self.context = []
        self._load()
    
    def _load(self):
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.context = data.get('context', [])
            except:
                pass
    
    def _save(self):
        try:
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump({'context': self.context[-self.max_size:]}, f, ensure_ascii=False)
        except:
            pass
    
    def add(self, entry: Dict):
        self.context.append(entry)
        if len(self.context) > self.max_size:
            self.context = self.context[-self.max_size:]
        self._save()
    
    def get_context(self, limit=3):
        return self.context[-limit:]
    
    def get_formatted_context(self, limit=3):
        context = self.get_context(limit)
        if not context:
            return "Нет истории"
        result = []
        for entry in context:
            role = entry.get('role', 'unknown')
            content = entry.get('content', entry.get('result', ''))
            if len(content) > 100:
                content = content[:100] + '...'
            result.append(f"{role}: {content}")
        return '\n'.join(result)
    
    def clear(self):
        self.context = []
        self._save()