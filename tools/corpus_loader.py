
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List

from config import Config


class CorpusLoader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = 0
        self._count_lines()

    def _count_lines(self):
        self.size = 0

        if not self.path.is_file():
            return

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
                errors="replace"
            ) as stream:
                for line in stream:
                    if self._parse_record(line) is not None:
                        self.size += 1

        except OSError:
            self.size = 0

    def _parse_record(self, line: str):
        if not isinstance(line, str):
            return None

        line = line.strip()

        if not line:
            return None

        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(record, dict):
            return None

        text = record.get("text")

        if not isinstance(text, str):
            return None

        text = text.strip()

        if len(text) < int(
            getattr(Config, "LIFE_MIN_ANSWER_FOR_PAIR", 60)
        ):
            return None

        title = record.get("title", "")

        if not isinstance(title, str):
            title = str(title)

        result = dict(record)
        result["title"] = title
        result["text"] = text

        return result

    def __iter__(self) -> Iterator[Dict[str, str]]:
        if not self.path.is_file():
            return

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
                errors="replace"
            ) as stream:
                for line in stream:
                    record = self._parse_record(line)

                    if record is not None:
                        yield record

        except OSError:
            return

    def __len__(self):
        return self.size

    def sample(
        self,
        n: int,
        seed: int = 42
    ) -> List[Dict[str, str]]:
        try:
            n = int(n)
        except (TypeError, ValueError):
            return []

        if n <= 0:
            return []

        rng = random.Random(seed)
        samples = []

        for index, record in enumerate(self):
            if len(samples) < n:
                samples.append(record)
                continue

            replacement = rng.randint(0, index)

            if replacement < n:
                samples[replacement] = record

        return samples
