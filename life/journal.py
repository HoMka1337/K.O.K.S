
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config


class Journal:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(
            path
            if path is not None
            else Config.LIFE_DIR / "journal.jsonl"
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        self._load()

    def _keep_limit(self) -> int:
        try:
            return max(
                1,
                int(Config.LIFE_JOURNAL_KEEP)
            )
        except (TypeError, ValueError):
            return 20000

    def _entry_limit(self) -> int:
        try:
            return max(
                1,
                int(Config.LIFE_JOURNAL_MAX_ENTRY_CHARS)
            )
        except (TypeError, ValueError):
            return 5000

    def _load(self) -> None:
        with self._lock:
            self.entries = []

            if not self.path.is_file():
                return

            keep = self._keep_limit()
            total = 0

            try:
                with self.path.open(
                    "r",
                    encoding="utf-8",
                    errors="replace"
                ) as stream:
                    for line in stream:
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(entry, dict):
                            continue

                        self.entries.append(entry)
                        total += 1

                        if len(self.entries) > keep:
                            del self.entries[
                                :len(self.entries) - keep
                            ]

            except OSError:
                self.entries = []
                return

            if total > keep:
                self._rewrite()

    def _rewrite(self) -> None:
        with self._lock:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            temporary = None

            try:
                fd, filename = tempfile.mkstemp(
                    prefix=".journal_",
                    suffix=".tmp",
                    dir=str(self.path.parent)
                )

                temporary = Path(filename)

                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8"
                ) as stream:
                    for entry in self.entries:
                        stream.write(
                            json.dumps(
                                entry,
                                ensure_ascii=False,
                                default=str
                            ) + "\n"
                        )

                    stream.flush()
                    os.fsync(stream.fileno())

                os.replace(
                    temporary,
                    self.path
                )

            finally:
                if temporary is not None:
                    temporary.unlink(
                        missing_ok=True
                    )

    def add(
        self,
        kind: str,
        content: str,
        **extra: Any
    ) -> Dict[str, Any]:
        if not isinstance(kind, str):
            kind = str(kind)

        if not isinstance(content, str):
            content = str(content)

        limit = self._entry_limit()

        entry = {
            "ts": time.time(),
            "kind": kind.strip()[:100],
            "content": content[:limit]
        }

        for key, value in extra.items():
            if isinstance(value, str):
                entry[key] = value[:limit]
            else:
                entry[key] = value

        with self._lock:
            self.entries.append(entry)

            keep = self._keep_limit()

            if len(self.entries) > keep:
                self.entries = self.entries[-keep:]
                self._rewrite()

            else:
                self.path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with self.path.open(
                    "a",
                    encoding="utf-8"
                ) as stream:
                    stream.write(
                        json.dumps(
                            entry,
                            ensure_ascii=False,
                            default=str
                        ) + "\n"
                    )

                    stream.flush()

        return entry

    def recent(
        self,
        limit: int = 20,
        kind: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return []

        if limit <= 0:
            return []

        with self._lock:
            if kind is None:
                items = self.entries
            else:
                items = [
                    entry
                    for entry in self.entries
                    if entry.get("kind") == kind
                ]

            return [
                dict(entry)
                for entry in items[-limit:]
            ]

    def last_successes(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self.recent(
            limit=limit,
            kind="learn"
        )

    def last_errors(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self.recent(
            limit=limit,
            kind="error"
        )

    def stats(self) -> Dict[str, int]:
        with self._lock:
            result = {}

            for entry in self.entries:
                kind = entry.get(
                    "kind",
                    "unknown"
                )

                if not isinstance(kind, str):
                    kind = str(kind)

                result[kind] = (
                    result.get(kind, 0) + 1
                )

            return result

    def as_training_pairs(self) -> List[str]:
        pairs = []
        seen = set()

        with self._lock:
            for entry in self.entries:
                if entry.get("kind") != "learn":
                    continue

                if entry.get("verified") is not True:
                    continue

                question = entry.get("question")
                answer = entry.get("answer")

                if not isinstance(question, str):
                    continue

                if not isinstance(answer, str):
                    continue

                question = question.strip()
                answer = answer.strip()

                if not question or not answer:
                    continue

                pair = f"{question}\n{answer}"

                if pair in seen:
                    continue

                seen.add(pair)
                pairs.append(pair)

        return pairs
