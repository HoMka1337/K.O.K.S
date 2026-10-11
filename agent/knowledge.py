import json
from pathlib import Path
from typing import Dict, List, Optional

from agent.semantics import (
    SemanticIndex,
    answer_looks_meaningful,
    clean_answer,
)


class Knowledge:
    def __init__(self, knowledge_path: Path):
        self.knowledge_path = knowledge_path
        self.entries: List[Dict[str, str]] = []
        self._index: Optional[SemanticIndex] = None
        self._load()

    def _load(self) -> None:
        if not self.knowledge_path.exists():
            return
        try:
            data = json.loads(self.knowledge_path.read_text(encoding='utf-8'))
            entries = data.get('entries', [])
            if isinstance(entries, list):
                self.entries = [e for e in entries if isinstance(e, dict)]
        except (OSError, json.JSONDecodeError):
            self.entries = []

    def _save(self) -> None:
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.write_text(
            json.dumps({'entries': self.entries}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _invalidate_index(self) -> None:
        self._index = None

    def _ensure_index(self) -> SemanticIndex:
        if self._index is None:
            docs = []
            for i, entry in enumerate(self.entries):
                text = ' '.join(
                    str(entry.get(k, '')) for k in ('content', 'source', 'url')
                )
                docs.append((str(i), text))
            self._index = SemanticIndex(docs)
        return self._index

    def add_url(self, url: str) -> None:
        if not any(entry.get('url') == url for entry in self.entries):
            self.entries.append({'type': 'url', 'url': url})
            self._save()
            self._invalidate_index()

    def add_text(self, text: str, source: str = 'manual') -> None:
        text = text.strip()
        if not text:
            return
        self.entries.append({'type': 'text', 'source': source, 'content': text})
        self._save()
        self._invalidate_index()

    def import_file(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            return f'Файл не найден: {path}'
        try:
            self.add_text(file_path.read_text(encoding='utf-8'), str(file_path))
            return f'Знания добавлены из файла: {path}'
        except (OSError, UnicodeError) as error:
            return f'Ошибка импорта знаний: {error}'

    def search(self, query: str, limit: int = 1) -> str:
        if not self.entries:
            return ''
        index = self._ensure_index()
        result = index.best(query, min_score=0.4)
        if result is None:
            return ''
        score, idx = result
        entry = self.entries[idx]
        content = entry.get('content') or entry.get('url') or ''
        content = clean_answer(content)
        if not answer_looks_meaningful(content):
            return ''
        return content

    def search_many(self, query: str, limit: int = 3) -> List[str]:
        if not self.entries:
            return []
        index = self._ensure_index()
        scored = []
        for i in range(len(self.entries)):
            s = index.score(query, i)
            if s > 0.3:
                scored.append((s, i))
        scored.sort(reverse=True)
        out = []
        for _, idx in scored[:limit]:
            entry = self.entries[idx]
            content = entry.get('content') or entry.get('url') or ''
            content = clean_answer(content)
            if answer_looks_meaningful(content):
                out.append(content)
        return out