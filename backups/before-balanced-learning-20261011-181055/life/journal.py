import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config


class Journal:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else (Config.LIFE_DIR / "journal.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            self.entries = []
        if len(self.entries) > Config.LIFE_JOURNAL_KEEP:
            self.entries = self.entries[-Config.LIFE_JOURNAL_KEEP:]
            self._rewrite()

    def _rewrite(self) -> None:
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                for entry in self.entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def add(self, kind: str, content: str, **extra: Any) -> Dict[str, Any]:
        entry = {
            "ts": time.time(),
            "kind": kind,          # goal / action / result / learn / error / thought
            "content": content,
        }
        entry.update(extra)
        self.entries.append(entry)
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return entry

    def recent(self, limit: int = 20, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        items = self.entries if kind is None else [e for e in self.entries if e.get("kind") == kind]
        return items[-limit:]

    def last_successes(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("kind") == "learn"][-limit:]

    def last_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("kind") == "error"][-limit:]

    def stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for entry in self.entries:
            kind = entry.get("kind", "unknown")
            stats[kind] = stats.get(kind, 0) + 1
        return stats

    def as_training_pairs(self) -> List[str]:
        pairs: List[str] = []
        for entry in self.entries:
            if entry.get("kind") != "learn":
                continue
            question = entry.get("question")
            answer = entry.get("answer")
            if question and answer:
                pairs.append(f"{question}\n{answer}")
        return pairs