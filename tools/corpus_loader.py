import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from config import Config


class CorpusLoader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = 0
        self._count_lines()

    def _count_lines(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for _ in f:
                    self.size += 1
        except OSError:
            pass

    def __iter__(self) -> Iterator[Dict[str, str]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if len(text) >= Config.LIFE_MIN_ANSWER_FOR_PAIR:
                    yield obj

    def sample(self, n: int, seed: int = 42) -> List[Dict[str, str]]:
        import random
        rng = random.Random(seed)
        if self.size <= n:
            return list(self)
        indices = set(rng.sample(range(self.size), n))
        out = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i in indices:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if len(out) >= n:
                    break
        return out