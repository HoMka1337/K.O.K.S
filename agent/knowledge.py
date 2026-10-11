
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

from agent.semantics import (
    SemanticIndex,
    answer_looks_meaningful,
    clean_answer,
)


class Knowledge:
    def __init__(self, knowledge_path: Path):
        self.knowledge_path = Path(knowledge_path)
        self.entries: List[Dict[str, str]] = []
        self._index: Optional[SemanticIndex] = None
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self._lock:
            self.entries = []
            self._index = None

            if not self.knowledge_path.is_file():
                return

            try:
                with self.knowledge_path.open(
                    "r",
                    encoding="utf-8"
                ) as stream:
                    data = json.load(stream)

                if not isinstance(data, dict):
                    return

                entries = data.get("entries", [])

                if not isinstance(entries, list):
                    return

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue

                    entry_type = entry.get("type")

                    if entry_type == "text":
                        content = entry.get("content", "")
                        source = entry.get("source", "manual")

                        if not isinstance(content, str):
                            continue

                        if not content.strip():
                            continue

                        self.entries.append({
                            "type": "text",
                            "source": str(source),
                            "content": content.strip()
                        })

                    elif entry_type == "url":
                        url = entry.get("url", "")

                        if not isinstance(url, str):
                            continue

                        if not url.strip():
                            continue

                        self.entries.append({
                            "type": "url",
                            "url": url.strip()
                        })

            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError
            ):
                self.entries = []

    def _save(self) -> None:
        with self._lock:
            self.knowledge_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            fd, temporary = tempfile.mkstemp(
                prefix=".knowledge_",
                suffix=".tmp",
                dir=str(self.knowledge_path.parent)
            )

            try:
                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8"
                ) as stream:
                    json.dump(
                        {"entries": self.entries},
                        stream,
                        ensure_ascii=False,
                        indent=2
                    )

                    stream.flush()
                    os.fsync(stream.fileno())

                os.replace(
                    temporary,
                    self.knowledge_path
                )

            finally:
                Path(temporary).unlink(
                    missing_ok=True
                )

    def _invalidate_index(self) -> None:
        with self._lock:
            self._index = None

    def _ensure_index(self) -> SemanticIndex:
        with self._lock:
            if self._index is None:
                documents = []

                for index, entry in enumerate(self.entries):
                    content = entry.get("content", "")
                    source = entry.get("source", "")
                    url = entry.get("url", "")

                    text = " ".join(
                        value
                        for value in (content, source, url)
                        if isinstance(value, str) and value
                    )

                    documents.append(
                        (str(index), text)
                    )

                self._index = SemanticIndex(documents)

            return self._index

    def add_url(self, url: str) -> None:
        if not isinstance(url, str):
            return

        url = url.strip()

        if not url:
            return

        with self._lock:
            for entry in self.entries:
                if entry.get("url") == url:
                    return

            self.entries.append({
                "type": "url",
                "url": url
            })

            self._save()
            self._invalidate_index()

    def add_text(
        self,
        text: str,
        source: str = "manual"
    ) -> None:
        if not isinstance(text, str):
            return

        text = text.strip()

        if not text:
            return

        if not isinstance(source, str):
            source = str(source)

        source = source.strip() or "manual"

        with self._lock:
            for entry in self.entries:
                if (
                    entry.get("type") == "text"
                    and entry.get("content") == text
                    and entry.get("source") == source
                ):
                    return

            self.entries.append({
                "type": "text",
                "source": source,
                "content": text
            })

            self._save()
            self._invalidate_index()

    def import_file(self, path: str) -> str:
        try:
            file_path = Path(path)
        except (TypeError, ValueError):
            return "Некорректный путь к файлу"

        if not file_path.is_file():
            return f"Файл не найден: {path}"

        try:
            text = file_path.read_text(
                encoding="utf-8"
            )

            if not text.strip():
                return f"Файл пустой: {path}"

            self.add_text(
                text,
                str(file_path)
            )

            return f"Знания добавлены из файла: {path}"

        except (OSError, UnicodeError, ValueError) as error:
            return f"Ошибка импорта знаний: {error}"

    def _entry_content(
        self,
        entry: Dict[str, str]
    ) -> str:
        content = entry.get("content", "")

        if not isinstance(content, str):
            return ""

        content = clean_answer(content)

        if not answer_looks_meaningful(content):
            return ""

        return content

    def search(
        self,
        query: str,
        limit: int = 1
    ) -> str:
        results = self.search_many(
            query,
            limit=limit
        )

        if not results:
            return ""

        return results[0]

    def search_many(
        self,
        query: str,
        limit: int = 3
    ) -> List[str]:
        if not isinstance(query, str):
            return []

        query = query.strip()

        if not query:
            return []

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return []

        if limit <= 0:
            return []

        with self._lock:
            if not self.entries:
                return []

            index = self._ensure_index()
            scored = []

            for position, entry in enumerate(self.entries):
                if entry.get("type") != "text":
                    continue

                score = index.score(
                    query,
                    position
                )

                if score > 0.3:
                    scored.append(
                        (score, position)
                    )

            scored.sort(
                key=lambda item: (
                    -item[0],
                    -item[1]
                )
            )

            results = []
            seen = set()

            for _, position in scored:
                content = self._entry_content(
                    self.entries[position]
                )

                if not content:
                    continue

                normalized = content.casefold()

                if normalized in seen:
                    continue

                seen.add(normalized)
                results.append(content)

                if len(results) >= limit:
                    break

            return results
