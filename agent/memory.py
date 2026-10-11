
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Union


class Memory:
    def __init__(
        self,
        memory_path: Union[str, Path] = "memory.json",
        max_size: int = 20
    ):
        self.memory_path = Path(memory_path)
        self.max_size = max(1, int(max_size))
        self.context = []
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        with self._lock:
            if not self.memory_path.is_file():
                self.context = []
                return

            try:
                with self.memory_path.open(
                    "r",
                    encoding="utf-8"
                ) as stream:
                    data = json.load(stream)

                if isinstance(data, dict):
                    entries = data.get("context", [])
                elif isinstance(data, list):
                    entries = data
                else:
                    entries = []

                if not isinstance(entries, list):
                    entries = []

                self.context = [
                    entry
                    for entry in entries
                    if isinstance(entry, dict)
                ][-self.max_size:]

            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError
            ):
                self.context = []

    def _save(self):
        with self._lock:
            self.memory_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            fd, temporary = tempfile.mkstemp(
                prefix=".memory_",
                suffix=".tmp",
                dir=str(self.memory_path.parent)
            )

            try:
                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8"
                ) as stream:
                    json.dump(
                        {
                            "context": self.context[
                                -self.max_size:
                            ]
                        },
                        stream,
                        ensure_ascii=False,
                        indent=2
                    )

                    stream.flush()
                    os.fsync(stream.fileno())

                os.replace(
                    temporary,
                    self.memory_path
                )

            finally:
                Path(temporary).unlink(
                    missing_ok=True
                )

    def add(self, entry: Dict):
        if not isinstance(entry, dict):
            return False

        role = entry.get("role", "unknown")
        content = entry.get(
            "content",
            entry.get("result", "")
        )

        if not isinstance(role, str):
            role = str(role)

        if not isinstance(content, str):
            content = str(content)

        if not content.strip():
            return False

        normalized_entry = dict(entry)
        normalized_entry["role"] = role
        normalized_entry["content"] = content

        with self._lock:
            self.context.append(normalized_entry)

            if len(self.context) > self.max_size:
                self.context = self.context[
                    -self.max_size:
                ]

            self._save()

        return True

    def get_context(self, limit=3):
        with self._lock:
            if limit is None:
                return [
                    dict(entry)
                    for entry in self.context
                ]

            limit = int(limit)

            if limit <= 0:
                return []

            return [
                dict(entry)
                for entry in self.context[-limit:]
            ]

    def get_formatted_context(self, limit=3):
        context = self.get_context(limit)

        if not context:
            return "Нет истории"

        result = []

        for entry in context:
            role = entry.get(
                "role",
                "unknown"
            )

            content = entry.get(
                "content",
                entry.get("result", "")
            )

            if not isinstance(role, str):
                role = str(role)

            if not isinstance(content, str):
                content = str(content)

            content = content.strip()

            if not content:
                continue

            if len(content) > 100:
                content = content[:100] + "..."

            result.append(
                f"{role}: {content}"
            )

        if not result:
            return "Нет истории"

        return "\n".join(result)

    def clear(self):
        with self._lock:
            self.context = []
            self._save()
